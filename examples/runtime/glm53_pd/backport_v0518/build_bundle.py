#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a local pd1 rollout bundle from verified operator inputs; never deploy."""

import argparse
import copy
import hashlib
import json
import re
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import yaml

PACKAGE = Path(__file__).resolve().parent
PATCH_NAME = re.compile(r"[0-9]{4}-[A-Za-z0-9][A-Za-z0-9_.-]*\.patch\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject YAML ambiguities before comparing deployment identities or data."""


def _unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class LiteralDumper(yaml.SafeDumper):
    pass


def _represent_string(dumper, value):
    return dumper.represent_scalar(
        "tag:yaml.org,2002:str", value, style="|" if "\n" in value else None
    )


LiteralDumper.add_representer(str, _represent_string)


def yaml_text(value):
    return yaml.dump(value, Dumper=LiteralDumper, sort_keys=False, allow_unicode=True)


def verified_inputs(archive_path, configmap_path, package=PACKAGE):
    expected = json.loads((package / "baseline_inputs.json").read_text())["files"]
    files = {}
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or stat.S_ISLNK(entry.external_attr >> 16)
            ):
                raise ValueError(f"Unsafe archive member: {entry.filename}")
            if entry.is_dir():
                continue
            if len(path.parts) != 2 or path.parts[0] != "patches":
                raise ValueError(f"Unexpected archive member: {entry.filename}")
            if path.name in files or path.name not in expected:
                raise ValueError(f"Duplicate or unexpected input: {path.name}")
            if entry.file_size > 2_000_000:
                raise ValueError(f"Oversized archive member: {path.name}")
            data = archive.read(entry)
            if sha256(data) != expected[path.name]:
                raise ValueError(f"Input hash mismatch: {path.name}")
            files[path.name] = data.decode("utf-8")
    if set(files) != set(expected):
        raise ValueError("The archive does not contain the recorded complete baseline")
    configmap = yaml.load(Path(configmap_path).read_text(), Loader=UniqueKeyLoader)
    if (
        not isinstance(configmap, dict)
        or configmap.get("apiVersion") != "v1"
        or configmap.get("kind") != "ConfigMap"
        or configmap.get("binaryData")
        or configmap.get("data") != files
    ):
        raise ValueError(
            "ConfigMap data must match every supplied baseline file exactly"
        )
    return files, configmap


def validate_stack(stack, baseline):
    if (
        stack.get("format_version") != 1
        or stack.get("base_revision") != baseline["upstream_revision"]
    ):
        raise ValueError("Unsupported stack format or upstream revision mismatch")
    entries = stack.get("patches")
    if not isinstance(entries, list) or not entries:
        raise ValueError("The stack must contain ordered patch entries")
    names, baseline_names, extension_names = [], [], []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Malformed stack entry")
        name, digest, stage = entry.get("name"), entry.get("sha256"), entry.get("stage")
        if not isinstance(name, str) or not PATCH_NAME.fullmatch(name):
            raise ValueError("Patch names must be safe numbered basenames")
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise ValueError(f"Invalid patch digest: {name}")
        if stage not in ("baseline", "extension"):
            raise ValueError(f"Invalid patch stage: {name}")
        names.append(name)
        (baseline_names if stage == "baseline" else extension_names).append(name)
    expected = sorted(name for name in baseline["files"] if name.endswith(".patch"))
    if (
        names != sorted(set(names))
        or baseline_names != expected
        or not extension_names
        or names != baseline_names + extension_names
    ):
        raise ValueError(
            "Stack must preserve the complete ordered baseline followed by unique extensions"
        )
    states = stack.get("states", {})
    if set(states) != {"upstream", "baseline", "candidate"}:
        raise ValueError("Stack must record all three source states")
    paths = set(states["upstream"])
    if not paths or any(set(state) != paths for state in states.values()):
        raise ValueError("Source states must cover the same affected paths")
    for path in paths:
        parts = PurePosixPath(path)
        if (
            parts.is_absolute()
            or ".." in parts.parts
            or not path.startswith("python/sglang/")
            or "\\" in path
        ):
            raise ValueError("Source state path must stay inside python/sglang")
        for state in states.values():
            digest = state[path]
            if digest is not None and (
                not isinstance(digest, str) or not SHA256.fullmatch(digest)
            ):
                raise ValueError(f"Invalid source digest: {path}")


def _role(rbg, name):
    found = [(i, r) for i, r in enumerate(rbg["spec"]["roles"]) if r["name"] == name]
    if len(found) != 1:
        raise ValueError(f"Expected one {name} role")
    index, role = found[0]
    template = role.get("standalonePattern", {}).get("template")
    if not isinstance(template, dict):
        raise ValueError(f"Expected an inline pod template for {name}")
    pointer = f"/spec/roles/{index}/standalonePattern/template"
    return index, role, template, pointer


def _identity_tests(rbg):
    return [
        {"op": "test", "path": "/metadata/name", "value": rbg["metadata"]["name"]},
        {
            "op": "test",
            "path": "/metadata/namespace",
            "value": rbg["metadata"]["namespace"],
        },
    ]


def require_fail_closed_startup(template, role_name):
    """The verified pd1 bash startup must not serve after an installer failure."""
    workers = [
        (index, container)
        for index, container in enumerate(template["spec"].get("containers", []))
        if len(command := container.get("command", [])) == 3
        and command[:2] == ["bash", "-c"]
        and isinstance(command[-1], str)
        and "exec sglang serve" in command[-1]
    ]
    if len(workers) != 1:
        raise ValueError(f"Expected the verified bash startup form on {role_name}")
    worker_index, worker = workers[0]
    lines = [
        line.strip()
        for line in worker["command"][-1].splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    bootstrap = "bash /sgl-workspace/sglang-fix/fix-patch.sh"
    if (
        len(lines) < 2
        or lines[0] != "set -euo pipefail"
        or lines[1] != bootstrap
        or lines.count(bootstrap) != 1
        or any(line.startswith("set ") for line in lines[1:])
    ):
        raise ValueError(
            f"Worker startup must retain fail-closed patch application on {role_name}"
        )
    start = next(
        (i for i, line in enumerate(lines) if line.startswith("exec sglang serve")),
        None,
    )
    if start is None or lines.index(bootstrap) >= start:
        raise ValueError(
            f"Patch application must precede worker startup on {role_name}"
        )
    return worker_index, worker


def runtime_patches(rbg, old_name, new_name):
    forward, rollback = _identity_tests(rbg), _identity_tests(rbg)
    candidate = copy.deepcopy(rbg)
    for name in ("prefill", "decode"):
        index, _, template, pointer = _role(rbg, name)
        worker_index, worker = require_fail_closed_startup(template, name)
        test_role = {"op": "test", "path": f"/spec/roles/{index}/name", "value": name}
        forward.append(test_role)
        rollback.append(copy.deepcopy(test_role))
        worker_path = pointer + f"/spec/containers/{worker_index}"
        startup_guards = [
            {"op": "test", "path": worker_path + "/name", "value": worker["name"]},
            {
                "op": "test",
                "path": worker_path + "/command",
                "value": worker["command"],
            },
        ]
        forward.extend(startup_guards)
        rollback.extend(copy.deepcopy(startup_guards))
        matches = [
            (i, v)
            for i, v in enumerate(template["spec"].get("volumes", []))
            if v.get("configMap", {}).get("name") == old_name
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one baseline ConfigMap volume on {name}")
        volume_index, _ = matches[0]
        path = pointer + f"/spec/volumes/{volume_index}/configMap/name"
        forward.extend(
            [
                {"op": "test", "path": path, "value": old_name},
                {"op": "replace", "path": path, "value": new_name},
            ]
        )
        rollback.extend(
            [
                {"op": "test", "path": path, "value": new_name},
                {"op": "replace", "path": path, "value": old_name},
            ]
        )
        _role(candidate, name)[2]["spec"]["volumes"][volume_index]["configMap"][
            "name"
        ] = new_name
    return forward, rollback, candidate


def router_candidates(rbg):
    """Independent experiments; neither is part of the runtime-only rollout."""
    index, _, template, pointer = _role(rbg, "router")
    containers = template["spec"]["containers"]
    matches = [
        (i, c)
        for i, c in enumerate(containers)
        if any(
            "sglang_router.launch_router" in str(arg) for arg in c.get("command", [])
        )
    ]
    if len(matches) != 1:
        raise ValueError("Expected one HTTP router command")
    container_index, container = matches[0]
    base = pointer + f"/spec/containers/{container_index}"
    guards = _identity_tests(rbg) + [
        {"op": "test", "path": f"/spec/roles/{index}/name", "value": "router"},
        {"op": "test", "path": base + "/name", "value": container["name"]},
    ]
    results = {}
    for i, command in enumerate(container.get("command", [])):
        if "sglang_router.launch_router" not in command:
            continue
        updated, count = re.subn(r"(?m)^[ \t]*--dp-aware[ \t]*\\?[ \t]*\n", "", command)
        if count != 1:
            raise ValueError("Expected one standalone --dp-aware line")
        path = base + f"/command/{i}"
        results["router-rank-selection"] = (
            guards
            + [
                {"op": "test", "path": path, "value": command},
                {"op": "replace", "path": path, "value": updated},
            ],
            guards
            + [
                {"op": "test", "path": path, "value": updated},
                {"op": "replace", "path": path, "value": command},
            ],
        )
    envs = [
        (i, e)
        for i, e in enumerate(container.get("env", []))
        if e.get("name") == "RUST_LOG"
    ]
    if len(envs) != 1 or not isinstance(envs[0][1].get("value"), str):
        raise ValueError("Expected one literal router RUST_LOG value")
    env_index, env = envs[0]
    path = base + f"/env/{env_index}/value"
    env_guard = {
        "op": "test",
        "path": base + f"/env/{env_index}/name",
        "value": "RUST_LOG",
    }
    updated = "info"
    results["router-log-level"] = (
        guards
        + [
            env_guard,
            {"op": "test", "path": path, "value": env["value"]},
            {"op": "replace", "path": path, "value": updated},
        ],
        guards
        + [
            env_guard,
            {"op": "test", "path": path, "value": updated},
            {"op": "replace", "path": path, "value": env["value"]},
        ],
    )
    return results


def build_bundle(
    archive_path, configmap_path, deployment_path, output_dir, *, package=PACKAGE
):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError(
            "Choose a new output directory; existing bundles are never overwritten"
        )
    files, configmap = verified_inputs(archive_path, configmap_path, package)
    documents = list(
        yaml.load_all(Path(deployment_path).read_text(), Loader=UniqueKeyLoader)
    )
    rbgs = [
        doc
        for doc in documents
        if isinstance(doc, dict) and doc.get("kind") == "RoleBasedGroup"
    ]
    if len(rbgs) != 1:
        raise ValueError("Expected one RoleBasedGroup input")
    rbg = rbgs[0]
    namespace = rbg["metadata"]["namespace"]
    if configmap["metadata"].get("namespace") != namespace:
        raise ValueError("ConfigMap and RBG must use the same namespace")
    stack = json.loads((package / "stack.json").read_text())
    validate_stack(stack, json.loads((package / "baseline_inputs.json").read_text()))
    for entry in stack["patches"]:
        if entry["stage"] == "extension":
            data = (package / "patches" / entry["name"]).read_bytes()
            if sha256(data) != entry["sha256"]:
                raise ValueError(f"Backport hash mismatch: {entry['name']}")
            files[entry["name"]] = data.decode("utf-8")
        elif (
            entry["name"] not in files
            or sha256(files[entry["name"]].encode()) != entry["sha256"]
        ):
            raise ValueError("Stack baseline differs from verified input")
    files["stack.json"] = json.dumps(stack, indent=2) + "\n"
    files["apply-stack.py"] = (package / "apply_stack.py").read_text()
    files["fix-patch.sh"] = (
        "#!/bin/bash\nset -euo pipefail\n"
        'patch_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
        'exec python3 "${patch_dir}/apply-stack.py" '
        '--source-dir /sgl-workspace/sglang --patch-dir "${patch_dir}"\n'
    )
    digest = sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode())
    old_name = configmap["metadata"]["name"]
    new_name = old_name[:220].rstrip("-.") + "-" + digest[:12]
    candidate_cm = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": new_name, "namespace": namespace},
        "immutable": True,
        "data": files,
    }
    data_bytes = sum(len(k.encode()) + len(v.encode()) for k, v in files.items())
    if data_bytes > 1_048_576:
        raise ValueError("Candidate ConfigMap exceeds the Kubernetes data size limit")
    forward, rollback, candidate_rbg = runtime_patches(rbg, old_name, new_name)
    experiments = router_candidates(rbg)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=".pd1-bundle-"
    ) as temporary:
        staging = Path(temporary)
        (staging / "patches").mkdir()
        for name, content in files.items():
            (staging / "patches" / name).write_text(content)
        (staging / "configmap.yaml").write_text(yaml_text(candidate_cm))
        (staging / "runtime-only.rbg.yaml").write_text(yaml_text(candidate_rbg))
        (staging / "baseline.rbg.yaml").write_text(yaml_text(rbg))
        outputs = {
            "runtime-only.patch.json": forward,
            "runtime-only.rollback.json": rollback,
        }
        for name, (change, revert) in experiments.items():
            outputs[name + ".patch.json"] = change
            outputs[name + ".rollback.json"] = revert
        for name, value in outputs.items():
            (staging / name).write_text(json.dumps(value, indent=2) + "\n")
        review = {
            "format_version": 1,
            "input_sha256": {
                Path(p).name: sha256(Path(p).read_bytes())
                for p in (archive_path, configmap_path, deployment_path)
            },
            "namespace": namespace,
            "rbg": rbg["metadata"]["name"],
            "baseline_configmap": old_name,
            "candidate_configmap": new_name,
            "patch_count": len(stack["patches"]),
            "configmap_data_bytes": data_bytes,
            "default_change": "prefill_and_decode_patch_volume_references_only",
            "separate_router_experiments": sorted(experiments),
            "service_documents_in_input": sum(
                isinstance(d, dict) and d.get("kind") == "Service" for d in documents
            ),
            "services_modified": False,
            "deployed": False,
            "prefill_decode_parameters_modified": False,
        }
        (staging / "review.json").write_text(json.dumps(review, indent=2) + "\n")
        # Renaming a complete new directory avoids exposing a half-built bundle.
        staging.rename(output_dir)
    return review


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--patches-zip", type=Path, required=True)
    parser.add_argument("--configmap", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build_bundle(
            args.patches_zip, args.configmap, args.deployment, args.output_dir
        )
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, yaml.YAMLError) as exc:
        parser.exit(1, f"Bundle build failed: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
