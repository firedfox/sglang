#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify and install the pinned pd1 Python patch stack, without starting a server.

Accept only a complete upstream, supplied baseline, or candidate source state.
GNU patch runs in a disposable shadow tree. Only verified final file contents
are installed in the source tree; a failed installation attempts rollback and
always exits unsuccessfully. Source symlinks are rejected. ConfigMap projection
symlinks in the read-only patch directory are supported.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_PATCH_NAME = re.compile(r"([0-9]{4})-[A-Za-z0-9_.-]+\.patch\Z")
_SOURCE_PATH = re.compile(r"python/sglang/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\Z")
_HUNK = re.compile(r"@@ -[0-9]+(?:,([0-9]+))? \+[0-9]+(?:,([0-9]+))? @@")
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


class StackError(RuntimeError):
    """An unrecognized state, invalid stack, or unsuccessful installation."""


class SourceFile(NamedTuple):
    data: bytes | None
    mode: int | None


def _digest(data: bytes | None) -> str | None:
    return hashlib.sha256(data).hexdigest() if data is not None else None


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise StackError(f"Duplicate manifest key: {key}")
        result[key] = value
    return result


def _source_path(value: str) -> str:
    if not isinstance(value, str) or not _SOURCE_PATH.fullmatch(value):
        raise StackError(f"Invalid source path: {value!r}")
    if any(part in (".", "..") for part in value.split("/")):
        raise StackError(f"Path traversal is not allowed: {value!r}")
    return value


def _header_path(value: str, prefix: str, allowed: set[str]) -> str | None:
    # GNU unified headers may have a tab-separated timestamp. Quoted Git paths
    # are parsed as one token; source filenames themselves use the strict
    # portable vocabulary checked by _source_path.
    value = value.split("\t", 1)[0]
    parts = shlex.split(value)
    if len(parts) != 1:
        raise StackError(f"Unsupported patch filename: {value!r}")
    value = parts[0]
    if value == "/dev/null":
        return None
    if not value.startswith(prefix + "/"):
        raise StackError(f"Patch path must start with {prefix}/: {value!r}")
    value = _source_path(value[2:])
    if value not in allowed:
        raise StackError(f"Patch modifies an unlisted source path: {value}")
    return value


def _validate_patch_paths(payload: bytes, allowed: set[str], name: str) -> None:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise StackError(f"Patch is not UTF-8 text: {name}") from exc
    files = 0
    old_left = new_left = 0
    in_file = False
    index = 0
    while index < len(lines):
        line = lines[index]
        if old_left or new_left:
            if line.startswith("\\ No newline at end of file"):
                index += 1
                continue
            if line.startswith(" "):
                old_left -= 1
                new_left -= 1
            elif line.startswith("-"):
                old_left -= 1
            elif line.startswith("+"):
                new_left -= 1
            else:
                raise StackError(f"Malformed unified hunk in {name}")
            if old_left < 0 or new_left < 0:
                raise StackError(f"Invalid unified hunk lengths in {name}")
        elif line.startswith("diff --git "):
            parts = shlex.split(line)
            if len(parts) != 4:
                raise StackError(f"Malformed Git diff header in {name}")
            _header_path(parts[2], "a", allowed)
            _header_path(parts[3], "b", allowed)
            in_file = False
        elif line.startswith("--- "):
            if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
                raise StackError(f"Missing unified new-file header in {name}")
            old_path = _header_path(line[4:], "a", allowed)
            new_path = _header_path(lines[index + 1][4:], "b", allowed)
            if old_path is None and new_path is None:
                raise StackError(f"Empty file change in {name}")
            if old_path is not None and new_path is not None and old_path != new_path:
                raise StackError(f"Renames are not supported in {name}")
            files += 1
            in_file = True
            index += 1
        elif line.startswith("@@"):
            match = _HUNK.match(line)
            if not in_file or match is None:
                raise StackError(f"Malformed unified hunk header in {name}")
            old_left = int(match[1]) if match[1] is not None else 1
            new_left = int(match[2]) if match[2] is not None else 1
        elif line.startswith(
            ("rename ", "copy ", "Index:", "GIT binary", "Binary files")
        ):
            raise StackError(f"Unsupported patch format in {name}")
        elif line.startswith(
            ("old mode ", "new mode ", "new file mode ", "deleted file mode ")
        ):
            if line.rsplit(" ", 1)[-1] not in ("100644", "100755"):
                raise StackError(f"Nonregular file mode in {name}")
        elif line.startswith("+++ "):
            raise StackError(f"Unpaired unified file header in {name}")
        index += 1
    if old_left or new_left or files == 0:
        raise StackError(f"Incomplete unified patch: {name}")


def _load_stack(patch_dir: Path):
    try:
        manifest = json.loads(
            (patch_dir / "stack.json").read_text(encoding="utf-8"),
            object_pairs_hook=_json_object,
        )
    except (OSError, ValueError) as exc:
        raise StackError(f"Cannot read stack.json: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("format_version")) is not int
        or manifest["format_version"] != 1
    ):
        raise StackError("Expected stack manifest format_version 1")
    if (
        not isinstance(manifest.get("base_revision"), str)
        or not manifest["base_revision"]
    ):
        raise StackError("Manifest requires a base_revision")
    states = manifest.get("states")
    if not isinstance(states, dict) or set(states) != {
        "upstream",
        "baseline",
        "candidate",
    }:
        raise StackError("Manifest must define upstream, baseline and candidate states")
    paths = None
    for name, values in states.items():
        if not isinstance(values, dict) or not values:
            raise StackError(f"Empty or invalid source state: {name}")
        for path, digest in values.items():
            _source_path(path)
            if digest is not None and (
                not isinstance(digest, str) or not _HASH.fullmatch(digest)
            ):
                raise StackError(f"Invalid source SHA-256 for {path}")
        if paths is None:
            paths = set(values)
        elif set(values) != paths:
            raise StackError("All source states must cover the same paths")
    records = manifest.get("patches")
    if not isinstance(records, list) or not records:
        raise StackError("Manifest must contain an ordered patch list")
    names, numbers, payloads = [], [], []
    extension_seen = False
    for record in records:
        if not isinstance(record, dict):
            raise StackError("Invalid patch entry")
        name = record.get("name")
        match = _PATCH_NAME.fullmatch(name) if isinstance(name, str) else None
        if match is None:
            raise StackError(f"Invalid patch basename: {name!r}")
        digest = record.get("sha256")
        if not isinstance(digest, str) or not _HASH.fullmatch(digest):
            raise StackError(f"Invalid patch SHA-256: {name}")
        stage = record.get("stage")
        if stage not in ("baseline", "extension") or (
            extension_seen and stage == "baseline"
        ):
            raise StackError("Baseline patches must precede extension patches")
        extension_seen |= stage == "extension"
        names.append(name)
        numbers.append(int(match[1]))
        payload = (patch_dir / name).read_bytes()
        if _digest(payload) != digest:
            raise StackError(f"Patch SHA-256 mismatch: {name}")
        _validate_patch_paths(payload, paths, name)
        payloads.append(payload)
    if (
        len(set(names)) != len(names)
        or len(set(numbers)) != len(numbers)
        or numbers != sorted(numbers)
    ):
        raise StackError("Patch names must have unique increasing numeric prefixes")
    actual = {path.name for path in patch_dir.iterdir() if path.name.endswith(".patch")}
    if actual != set(names):
        raise StackError(
            f"Patch list mismatch; extra={sorted(actual - set(names))}, missing={sorted(set(names) - actual)}"
        )
    return manifest, payloads


@contextmanager
def _parent_fd(root_fd: int, relative: str, *, create=False, created=None):
    fd = os.dup(root_fd)
    components = relative.split("/")
    try:
        for index, component in enumerate(components[:-1]):
            try:
                child = os.open(component, _DIR_FLAGS, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    yield None, components[-1]
                    return
                os.mkdir(component, 0o755, dir_fd=fd)
                if created is not None:
                    created.append("/".join(components[: index + 1]))
                child = os.open(component, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd, components[-1]
    except OSError as exc:
        raise StackError(
            f"Cannot safely access {relative}; source symlinks are not supported: {exc}"
        ) from exc
    finally:
        os.close(fd)


def _read_file(root_fd: int, relative: str) -> SourceFile:
    with _parent_fd(root_fd, relative) as (parent, name):
        if parent is None:
            return SourceFile(None, None)
        try:
            # Nonblocking open lets us reject FIFOs before attempting a read.
            fd = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
            )
        except FileNotFoundError:
            return SourceFile(None, None)
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise StackError(f"Source must be a regular file: {relative}")
            return SourceFile(handle.read(), stat.S_IMODE(info.st_mode))


def _snapshot(root_fd: int, paths) -> dict[str, SourceFile]:
    return {path: _read_file(root_fd, path) for path in sorted(paths)}


def _hashes(snapshot) -> dict[str, str | None]:
    return {path: _digest(info.data) for path, info in snapshot.items()}


def _preflight(original, expected, records, payloads):
    executable = shutil.which("patch")
    if executable is None:
        raise StackError("GNU patch is required")
    version = subprocess.run(
        [executable, "--version"], capture_output=True, text=True, check=False
    )
    if version.returncode or not version.stdout.startswith("GNU patch"):
        raise StackError("The patch executable must be GNU patch")
    with tempfile.TemporaryDirectory(prefix="sglang-pd1-stack-") as temporary:
        directory = Path(temporary)
        shadow = directory / "source"
        shadow.mkdir()
        for relative, info in original.items():
            if info.data is not None:
                target = shadow / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(info.data)
        for record, payload in zip(records, payloads, strict=True):
            patch_file = directory / record["name"]
            patch_file.write_bytes(payload)
            result = subprocess.run(
                [
                    executable,
                    "-p1",
                    "--unified",
                    "--forward",
                    "--batch",
                    "--fuzz=0",
                    "--get=0",
                    "--no-backup-if-mismatch",
                    "--reject-file=-",
                    "--input",
                    str(patch_file),
                ],
                cwd=shadow,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                raise StackError(
                    f"Preflight failed for {record['name']}: {(result.stdout + result.stderr)[-4000:]}"
                )
        files = {
            str(path.relative_to(shadow))
            for path in shadow.rglob("*")
            if not path.is_dir()
        }
        if files - set(expected):
            raise StackError("Preflight created unlisted files")
        fd = os.open(shadow, _DIR_FLAGS)
        try:
            final = _snapshot(fd, expected)
        finally:
            os.close(fd)
        if _hashes(final) != expected:
            raise StackError("Preflight final source SHA-256 does not match candidate")
        return final


def _install_file(root_fd: int, relative: str, info: SourceFile, created) -> None:
    with _parent_fd(
        root_fd, relative, create=info.data is not None, created=created
    ) as (parent, name):
        if parent is None:
            return
        if info.data is None:
            try:
                os.unlink(name, dir_fd=parent)
            except FileNotFoundError:
                pass
            return
        temporary = f".sglang-pd1-{uuid.uuid4().hex}.tmp"
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent,
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(info.data)
                os.fchmod(
                    handle.fileno(), info.mode if info.mode is not None else 0o644
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
        finally:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass


def _install(root_fd, original, final):
    attempted, created = [], []
    try:
        for relative, before in original.items():
            if before.data == final[relative].data:
                continue
            if _read_file(root_fd, relative) != before:
                raise StackError(f"Source changed during installation: {relative}")
            attempted.append(relative)
            _install_file(
                root_fd,
                relative,
                SourceFile(final[relative].data, before.mode),
                created,
            )
        if _hashes(_snapshot(root_fd, final)) != _hashes(final):
            raise StackError("Installed source does not match the verified candidate")
    except BaseException as exc:
        failed = []
        for relative in reversed(attempted):
            try:
                current = _read_file(root_fd, relative)
                if current == original[relative]:
                    continue
                if current.data != final[relative].data:
                    raise StackError("File changed after installation")
                _install_file(root_fd, relative, original[relative], created=[])
            except BaseException:
                failed.append(relative)
        for relative in reversed(created):
            try:
                with _parent_fd(root_fd, relative) as (parent, name):
                    if parent is not None:
                        os.rmdir(name, dir_fd=parent)
            except (OSError, StackError):
                pass
        outcome = (
            f"rollback incomplete for {failed}"
            if failed
            else "original affected files restored"
        )
        raise StackError(f"Installation failed; {outcome}: {exc}") from exc


def apply_stack(source_dir, patch_dir, *, dry_run=False) -> dict:
    source = Path(source_dir).resolve(strict=True)
    patch_dir = Path(patch_dir).resolve(strict=True)
    manifest, payloads = _load_stack(patch_dir)
    fd = os.open(source, _DIR_FLAGS)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StackError(
                "Another installer holds the source directory lock"
            ) from exc
        states = manifest["states"]
        original = _snapshot(fd, states["upstream"])
        observed = _hashes(original)
        state = next(
            (
                name
                for name in ("candidate", "baseline", "upstream")
                if observed == states[name]
            ),
            None,
        )
        if state is None:
            raise StackError(
                "Unknown or partially patched source state; refusing to modify it"
            )
        indices = [
            index
            for index, record in enumerate(manifest["patches"])
            if state == "upstream"
            or (state == "baseline" and record["stage"] == "extension")
        ]
        records = [manifest["patches"][index] for index in indices]
        result = {
            "state": state,
            "final_state": state,
            "applied": [],
            "planned": [record["name"] for record in records],
            "noop": state == "candidate",
            "dry_run": bool(dry_run),
            "validated_patches": len(payloads),
        }
        if state == "candidate":
            return result
        final = _preflight(
            original,
            states["candidate"],
            records,
            [payloads[index] for index in indices],
        )
        if _snapshot(fd, original) != original:
            raise StackError("Source changed during preflight; refusing to install")
        if not dry_run:
            _install(fd, original, final)
            result["applied"] = result["planned"]
            result["final_state"] = "candidate"
        return result
    finally:
        os.close(fd)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--patch-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = apply_stack(args.source_dir, args.patch_dir, dry_run=args.dry_run)
    except (StackError, OSError, ValueError) as exc:
        print(f"[patch-stack] ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
