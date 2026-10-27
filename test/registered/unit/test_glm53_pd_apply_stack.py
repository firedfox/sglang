"""CPU regression checks for the pd1 source-state patch installer.

Tests invoke real GNU patch against disposable files; no SGLang import, model,
GPU, network, Kubernetes API or source checkout mutation is required.
"""

import copy
import difflib
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import runpy
import shutil
import stat
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "examples/runtime/glm53_pd/backport_v0518/apply_stack.py"
runpy.run_path(str(REPO_ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
](est_time=3, suite="base-a-test-cpu")
spec = importlib.util.spec_from_file_location("glm53_pd_apply_stack", SCRIPT)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def digest(data):
    return hashlib.sha256(data).hexdigest() if data is not None else None


def unified_patch(before, after):
    pieces = []
    for path in sorted(before):
        if before[path] == after[path]:
            continue
        pieces.extend(
            difflib.unified_diff(
                (before[path] or b"").decode().splitlines(keepends=True),
                (after[path] or b"").decode().splitlines(keepends=True),
                fromfile="a/" + path if before[path] is not None else "/dev/null",
                tofile="b/" + path if after[path] is not None else "/dev/null",
            )
        )
    return "".join(pieces).encode()


class TestGLM53PDApplyStack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        executable = shutil.which("patch")
        if executable is None:
            raise unittest.SkipTest("GNU patch is required")
        version = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, check=False
        )
        if version.returncode or not version.stdout.startswith("GNU patch"):
            raise unittest.SkipTest("GNU patch is required")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.patches = self.root / "patches"
        self.source.mkdir()
        self.patches.mkdir()
        self.new = "python/sglang/a_new/resource.py"
        self.obsolete = "python/sglang/b_obsolete.py"
        self.runtime = "python/sglang/z_runtime.py"
        self.states = {
            "upstream": {
                self.new: None,
                self.obsolete: b"old_resource = 1\n",
                self.runtime: b"release_counter = 0\n",
            },
            "baseline": {
                self.new: b"queue = 1\n",
                self.obsolete: b"old_resource = 1\n",
                self.runtime: b"release_counter = 1\n",
            },
            "candidate": {
                self.new: b"queue = 2\n",
                self.obsolete: None,
                self.runtime: b"release_counter = 2\n",
            },
        }
        self.names = ["0003-baseline.patch", "0042-feedback.patch"]
        self.manifest = {
            "format_version": 1,
            "base_revision": "a" * 40,
            "patches": [],
            "states": {
                name: {path: digest(data) for path, data in files.items()}
                for name, files in self.states.items()
            },
        }
        for name, before, after, stage in zip(
            self.names,
            ("upstream", "baseline"),
            ("baseline", "candidate"),
            ("baseline", "extension"),
        ):
            payload = unified_patch(self.states[before], self.states[after])
            (self.patches / name).write_bytes(payload)
            self.manifest["patches"].append(
                {"name": name, "sha256": digest(payload), "stage": stage}
            )
        self.write_manifest()
        self.write_state("upstream")
        (self.source / "README").write_bytes(b"unrelated file\n")

    def write_manifest(self):
        (self.patches / "stack.json").write_text(json.dumps(self.manifest))

    def write_state(self, state):
        for relative, data in self.states[state].items():
            path = self.source / relative
            if data is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                path.chmod(0o640)

    def replace_patch(self, index, payload):
        (self.patches / self.names[index]).write_bytes(payload)
        self.manifest["patches"][index]["sha256"] = digest(payload)
        self.write_manifest()

    def source_snapshot(self):
        return {
            str(path.relative_to(self.source)): (
                path.read_bytes(),
                stat.S_IMODE(path.stat().st_mode),
            )
            for path in self.source.rglob("*")
            if path.is_file()
        }

    def assert_state(self, name):
        for relative, expected in self.states[name].items():
            path = self.source / relative
            self.assertEqual(path.read_bytes() if path.exists() else None, expected)
        self.assertEqual((self.source / "README").read_bytes(), b"unrelated file\n")

    def assert_rejected_unchanged(self, pattern):
        before = self.source_snapshot()
        with self.assertRaisesRegex(installer.StackError, pattern):
            installer.apply_stack(self.source, self.patches)
        self.assertEqual(self.source_snapshot(), before)

    def test_upstream_installs_full_stack_and_preserves_file_modes(self):
        result = installer.apply_stack(self.source, self.patches)
        self.assertEqual(result["state"], "upstream")
        self.assertEqual(result["final_state"], "candidate")
        self.assertEqual(result["applied"], self.names)
        self.assertFalse(result["noop"])
        self.assert_state("candidate")
        self.assertEqual(
            stat.S_IMODE((self.source / self.runtime).stat().st_mode), 0o640
        )
        self.assertEqual(stat.S_IMODE((self.source / self.new).stat().st_mode), 0o644)

    def test_baseline_only_installs_extensions_then_noop(self):
        self.write_state("baseline")
        result = installer.apply_stack(self.source, self.patches)
        self.assertEqual(result["state"], "baseline")
        self.assertEqual(result["applied"], self.names[1:])
        self.assert_state("candidate")
        files = [path for path in self.source.rglob("*") if path.is_file()]
        before = {path: path.stat() for path in files}
        result = installer.apply_stack(self.source, self.patches)
        self.assertEqual(result["state"], "candidate")
        self.assertTrue(result["noop"])
        self.assertEqual(result["planned"], [])
        self.assertEqual(result["applied"], [])
        self.assertEqual({path: path.stat() for path in files}, before)

    def test_dry_run_preflights_without_installing(self):
        before = self.source_snapshot()
        result = installer.apply_stack(self.source, self.patches, dry_run=True)
        self.assertEqual(result["planned"], self.names)
        self.assertEqual(result["applied"], [])
        self.assertEqual(result["final_state"], "upstream")
        self.assertTrue(result["dry_run"])
        self.assertFalse(result["noop"])
        self.assertEqual(self.source_snapshot(), before)
        self.assertFalse((self.source / "python/sglang/a_new").exists())

    def test_patch_tampering_rejected_even_for_candidate(self):
        for state in ("upstream", "baseline", "candidate"):
            with self.subTest(state=state):
                self.write_state(state)
                path = self.patches / self.names[0]
                original = path.read_bytes()
                path.write_bytes(original + b"tampered\n")
                self.assert_rejected_unchanged("SHA-256 mismatch")
                path.write_bytes(original)

    def test_unknown_and_partial_source_rejected(self):
        for data in (b"locally modified\n", self.states["candidate"][self.runtime]):
            with self.subTest(data=data):
                (self.source / self.runtime).write_bytes(data)
                self.assert_rejected_unchanged("Unknown or partially patched")

    def test_conflict_in_later_patch_does_not_apply_earlier_patch_to_source(self):
        payload = (self.patches / self.names[1]).read_bytes()
        payload = payload.replace(b"-release_counter = 1", b"-wrong_source = 1")
        self.replace_patch(1, payload)
        self.assert_rejected_unchanged("Preflight failed")
        self.assertFalse((self.source / "python/sglang/a_new").exists())

    def test_candidate_hash_mismatch_does_not_install(self):
        self.manifest["states"]["candidate"][self.runtime] = "0" * 64
        self.write_manifest()
        self.assert_rejected_unchanged("Preflight final source SHA-256")

    def test_manifest_traversal_absolute_and_outside_namespace_rejected(self):
        original = copy.deepcopy(self.manifest)
        for path in (
            "python/sglang/../../outside.py",
            "python/sglang/./runtime.py",
            "/python/sglang/runtime.py",
            "other/runtime.py",
        ):
            with self.subTest(path=path):
                self.manifest = copy.deepcopy(original)
                for state in self.manifest["states"].values():
                    state[path] = state.pop(self.runtime)
                self.write_manifest()
                self.assert_rejected_unchanged("Invalid source path|Path traversal")

    def test_patch_traversal_or_unlisted_target_rejected(self):
        original = (self.patches / self.names[0]).read_bytes()
        for path in ("python/sglang/../../outside.py", "python/sglang/unlisted.py"):
            with self.subTest(path=path):
                payload = original.replace(self.runtime.encode(), path.encode())
                self.replace_patch(0, payload)
                self.assert_rejected_unchanged("Path traversal|unlisted source path")

    def test_git_symlink_mode_and_incomplete_hunk_rejected(self):
        original = (self.patches / self.names[0]).read_bytes()
        self.replace_patch(0, b"new file mode 120000\n" + original)
        self.assert_rejected_unchanged("Nonregular file mode")
        self.replace_patch(0, original.rsplit(b"\n", 2)[0] + b"\n")
        self.assert_rejected_unchanged("Incomplete unified patch")

    def test_extra_patch_rejected(self):
        (self.patches / "0099-unlisted.patch").write_bytes(b"unlisted\n")
        self.assert_rejected_unchanged("Patch list mismatch")

    def test_nonincreasing_or_duplicate_numeric_order_rejected(self):
        original = copy.deepcopy(self.manifest)
        self.manifest["patches"].reverse()
        # Keep stages valid to exercise the independent numeric-order check.
        self.manifest["patches"][0]["stage"] = "baseline"
        self.manifest["patches"][1]["stage"] = "extension"
        self.write_manifest()
        self.assert_rejected_unchanged("unique increasing numeric prefixes")
        self.manifest = original
        renamed = "0003-duplicate.patch"
        (self.patches / self.names[1]).rename(self.patches / renamed)
        self.manifest["patches"][1]["name"] = renamed
        self.write_manifest()
        self.assert_rejected_unchanged("unique increasing numeric prefixes")

    def test_stage_order_rejected(self):
        self.manifest["patches"][0]["stage"] = "extension"
        self.manifest["patches"][1]["stage"] = "baseline"
        self.write_manifest()
        self.assert_rejected_unchanged("Baseline patches must precede")

    def test_manifest_duplicate_keys_rejected(self):
        raw = (self.patches / "stack.json").read_text()
        (self.patches / "stack.json").write_text(
            raw.replace(
                '"format_version": 1', '"format_version": 1, "format_version": 1'
            )
        )
        self.assert_rejected_unchanged("Duplicate manifest key")

    def test_manifest_boolean_version_rejected(self):
        self.manifest["format_version"] = True
        self.write_manifest()
        self.assert_rejected_unchanged("format_version 1")

    def test_configmap_projected_symlinks_supported(self):
        generation = self.patches / "..2026_generation"
        generation.mkdir()
        (self.patches / "..data").symlink_to(generation.name, target_is_directory=True)
        for name in ["stack.json", *self.names]:
            (self.patches / name).rename(generation / name)
            (self.patches / name).symlink_to("..data/" + name)
        result = installer.apply_stack(self.source, self.patches)
        self.assertEqual(result["applied"], self.names)
        self.assert_state("candidate")

    def test_source_file_symlink_rejected_even_if_contents_match(self):
        for external in (True, False):
            with self.subTest(external=external):
                target = (self.root if external else self.source) / "link-target.py"
                target.write_bytes(self.states["upstream"][self.runtime])
                path = self.source / self.runtime
                path.unlink()
                path.symlink_to(target)
                with self.assertRaisesRegex(installer.StackError, "symlinks"):
                    installer.apply_stack(self.source, self.patches)
                self.assertEqual(
                    target.read_bytes(), self.states["upstream"][self.runtime]
                )
                self.assertTrue(path.is_symlink())
                path.unlink()
                path.write_bytes(self.states["upstream"][self.runtime])

    def test_source_parent_symlink_rejected(self):
        outside = self.root / "outside"
        (self.source / "python/sglang").rename(outside)
        (self.source / "python/sglang").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(installer.StackError, "symlinks"):
            installer.apply_stack(self.source, self.patches)
        self.assertEqual(
            (outside / "z_runtime.py").read_bytes(),
            self.states["upstream"][self.runtime],
        )
        self.assertFalse((outside / "a_new").exists())

    def test_fifo_source_rejected_without_waiting_for_writer(self):
        path = self.source / self.runtime
        path.unlink()
        os.mkfifo(path)
        with self.assertRaisesRegex(installer.StackError, "regular file"):
            installer.apply_stack(self.source, self.patches)

    def test_source_directory_lock_prevents_overlapping_install(self):
        # Directory locking avoids adding a lock file to the source tree.
        fd = os.open(self.source, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.assert_rejected_unchanged("directory lock")

    def test_install_failure_rolls_back_created_and_deleted_files(self):
        before = self.source_snapshot()
        install_file = installer._install_file
        failed = False

        def fail_once(root_fd, relative, info, created):
            nonlocal failed
            if relative == self.runtime and not failed:
                failed = True
                raise OSError("injected disk failure")
            return install_file(root_fd, relative, info, created)

        with mock.patch.object(installer, "_install_file", side_effect=fail_once):
            with self.assertRaisesRegex(
                installer.StackError, "original affected files restored"
            ):
                installer.apply_stack(self.source, self.patches)
        self.assertEqual(self.source_snapshot(), before)
        self.assertFalse((self.source / "python/sglang/a_new").exists())

    def test_failure_after_atomic_replace_also_rolls_back(self):
        before = self.source_snapshot()
        install_file = installer._install_file
        failed = False

        def fail_after_replace(root_fd, relative, info, created):
            nonlocal failed
            install_file(root_fd, relative, info, created)
            if not failed:
                failed = True
                raise OSError("injected post-rename failure")

        with mock.patch.object(
            installer, "_install_file", side_effect=fail_after_replace
        ):
            with self.assertRaisesRegex(
                installer.StackError, "original affected files restored"
            ):
                installer.apply_stack(self.source, self.patches)
        self.assertEqual(self.source_snapshot(), before)
        self.assertFalse((self.source / "python/sglang/a_new").exists())

    def test_incomplete_rollback_is_reported_without_clobbering_external_change(self):
        install_file = installer._install_file
        external_change = b"modified by another process after replacement\n"

        def interrupt_after_other_writer(root_fd, relative, info, created):
            if relative == self.runtime:
                (self.source / self.new).write_bytes(external_change)
                raise OSError("injected install failure")
            return install_file(root_fd, relative, info, created)

        with mock.patch.object(
            installer, "_install_file", side_effect=interrupt_after_other_writer
        ):
            with self.assertRaisesRegex(installer.StackError, "rollback incomplete"):
                installer.apply_stack(self.source, self.patches)
        self.assertEqual((self.source / self.new).read_bytes(), external_change)
        self.assertEqual(
            (self.source / self.obsolete).read_bytes(),
            self.states["upstream"][self.obsolete],
        )
        self.assertEqual(
            (self.source / self.runtime).read_bytes(),
            self.states["upstream"][self.runtime],
        )
        self.assert_rejected_unchanged("Unknown or partially patched")

    def test_concurrent_source_change_after_preflight_is_not_overwritten(self):
        preflight = installer._preflight
        modified = b"another process changed this file\n"

        def change_during_preflight(*args):
            final = preflight(*args)
            (self.source / self.runtime).write_bytes(modified)
            return final

        with mock.patch.object(
            installer, "_preflight", side_effect=change_during_preflight
        ):
            with self.assertRaisesRegex(
                installer.StackError, "Source changed during preflight"
            ):
                installer.apply_stack(self.source, self.patches)
        self.assertEqual((self.source / self.runtime).read_bytes(), modified)
        self.assertFalse((self.source / "python/sglang/a_new").exists())
        self.assertTrue((self.source / self.obsolete).exists())

    def test_cli_dry_run_returns_json_and_errors_return_nonzero(self):
        args = ["--source-dir", str(self.source), "--patch-dir", str(self.patches)]
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(installer.main([*args, "--dry-run"]), 0)
        self.assertEqual(json.loads(output.getvalue())["applied"], [])
        (self.source / self.runtime).write_bytes(b"unknown\n")
        errors = io.StringIO()
        with redirect_stderr(errors):
            self.assertEqual(installer.main(args), 1)
        self.assertIn("Unknown or partially patched", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
