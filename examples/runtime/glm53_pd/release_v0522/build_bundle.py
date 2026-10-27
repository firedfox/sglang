#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export the reviewed release migration as an exact-source startup patch.

This builds artifacts only. It never pulls an image or changes a cluster.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

PACKAGE = Path(__file__).resolve().parent
BASE_REVISION = "baa9befa381511cd4d56f83980951e5a45fffc6d"
BASE_REF = "release/v0.5.22"
OLD_FEATURE_REVISION = "ed565e76dd4aed44ab3f4141fae3450dba3b0bc2"
TREE_CORE_ENV = "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND"
REQUIRED_WORKER_ENV = {TREE_CORE_ENV: "python"}

_spec = importlib.util.spec_from_file_location(
    "_glm53_pd_bundle_helpers", PACKAGE.parent / "backport_v0518/build_bundle.py"
)
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)


def git(source_dir, *args):
    return subprocess.check_output(["git", *args], cwd=source_dir)


def digest(data):
    return hashlib.sha256(data).hexdigest() if data is not None else None


def blob(source_dir, revision, path):
    entries = git(source_dir, "ls-tree", revision, "--", path).splitlines()
    if not entries:
        return None
    if len(entries) != 1 or entries[0].split()[0] not in (b"100644", b"100755"):
        raise ValueError(f"Expected a regular source file: {path}")
    return git(source_dir, "show", f"{revision}:{path}")


def export_source(source_dir, source_ref, output_dir, *, base_revision=BASE_REVISION):
    """Export committed Python/Gateway changes; never silently omit local edits."""
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    revision = (
        git(source_dir, "rev-parse", "--verify", source_ref + "^{commit}")
        .decode()
        .strip()
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", base_revision, revision],
        cwd=source_dir,
        check=True,
    )
    if git(
        source_dir,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "python/sglang",
        "sgl-model-gateway",
    ).strip():
        raise ValueError("Commit all runtime and Gateway changes before exporting")
    paths = [
        p.decode()
        for p in git(
            source_dir,
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            base_revision,
            revision,
            "--",
            "python/sglang",
        ).split(b"\0")
        if p
    ]
    if not paths:
        raise ValueError("The selected revision contains no Python migration")
    # Also identify the release even when this file has no custom changes.
    version_path = "python/sglang/version.py"
    if blob(source_dir, base_revision, version_path) is not None:
        paths = sorted(set(paths) | {version_path})
    upstream = {p: digest(blob(source_dir, base_revision, p)) for p in paths}
    candidate = {p: digest(blob(source_dir, revision, p)) for p in paths}
    payload = git(
        source_dir,
        "diff",
        "--no-ext-diff",
        "--no-renames",
        "--full-index",
        base_revision,
        revision,
        "--",
        "python/sglang",
    )
    payload.decode("utf-8")
    name = "0001-glm53-pd-runtime.patch"
    stack = {
        "format_version": 1,
        "base_revision": base_revision,
        "source_revision": revision,
        "states": {"upstream": upstream, "baseline": upstream, "candidate": candidate},
        "patches": [{"name": name, "sha256": digest(payload), "stage": "extension"}],
    }
    patches = output_dir / "patches"
    patches.mkdir(parents=True)
    (patches / name).write_bytes(payload)
    (patches / "stack.json").write_text(json.dumps(stack, indent=2) + "\n")
    shutil.copyfile(
        PACKAGE.parent / "backport_v0518/apply_stack.py", patches / "apply-stack.py"
    )
    shutil.copyfile(PACKAGE / "check_source.py", patches / "check-source.py")
    shutil.copyfile(PACKAGE / "fix-patch.sh", patches / "fix-patch.sh")
    gateway = git(
        source_dir,
        "diff",
        "--no-ext-diff",
        "--no-renames",
        "--full-index",
        base_revision,
        revision,
        "--",
        "sgl-model-gateway",
    )
    gateway.decode("utf-8")
    (output_dir / "gateway.patch").write_bytes(gateway)
    manifest = {
        "format_version": 1,
        "base_ref": BASE_REF,
        "base_revision": base_revision,
        "base_tag": "v0.5.22",
        "source_revision": revision,
        "previous_feature_revision": OLD_FEATURE_REVISION,
        "python_source_paths": len(paths),
        "runtime_patch_sha256": digest(payload),
        "gateway_patch_sha256": digest(gateway),
        "gateway_requires_rebuild": bool(gateway),
        "required_worker_env": REQUIRED_WORKER_ENV,
        "image_contents_inspected": False,
        "gpu_validated": False,
        "deployed": False,
    }
    (output_dir / "migration-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def deployment_patches(rbg, new_name, image):
    """Guard worker images, patch volumes and the required Python tree profile."""
    forward, rollback = helpers._identity_tests(rbg), helpers._identity_tests(rbg)
    candidate = copy.deepcopy(rbg)
    old_values = {}
    for role_name in ("prefill", "decode"):
        index, _, template, pointer = helpers._role(rbg, role_name)
        worker_index, worker = helpers.require_fail_closed_startup(template, role_name)
        worker_path = pointer + f"/spec/containers/{worker_index}"
        guards = [
            {"op": "test", "path": f"/spec/roles/{index}/name", "value": role_name},
            {"op": "test", "path": worker_path + "/name", "value": worker["name"]},
            {
                "op": "test",
                "path": worker_path + "/command",
                "value": worker["command"],
            },
            {
                "op": "test",
                "path": worker_path + "/volumeMounts",
                "value": worker["volumeMounts"],
            },
        ]
        forward.extend(guards)
        rollback.extend(copy.deepcopy(guards))
        # Release defaults to a compiled Rust tree. Two migrated fixes execute in
        # Python UnifiedTreeCore, so select it in this deployment profile rather
        # than silently changing the upstream global default.
        old_env = worker.get("env")
        if old_env is not None and not isinstance(old_env, list):
            raise ValueError(f"Expected a list of environment variables on {role_name}")
        new_env = copy.deepcopy(old_env or [])
        matches = [
            i for i, value in enumerate(new_env) if value.get("name") == TREE_CORE_ENV
        ]
        if len(matches) > 1:
            raise ValueError(f"Duplicate {TREE_CORE_ENV} on {role_name}")
        required = {"name": TREE_CORE_ENV, "value": "python"}
        if matches:
            new_env[matches[0]] = required
        else:
            new_env.append(required)
        env_path = worker_path + "/env"
        if "env" in worker:
            forward.append({"op": "test", "path": env_path, "value": old_env})
            forward.append({"op": "replace", "path": env_path, "value": new_env})
            rollback.append({"op": "test", "path": env_path, "value": new_env})
            rollback.append({"op": "replace", "path": env_path, "value": old_env})
        else:
            # JSON Patch has no test-for-absence operation. A complete container
            # guard prevents overwriting an env list added after export.
            forward.append(
                {"op": "test", "path": worker_path, "value": copy.deepcopy(worker)}
            )
            forward.append({"op": "add", "path": env_path, "value": new_env})
            rollback.append({"op": "test", "path": env_path, "value": new_env})
            rollback.append({"op": "remove", "path": env_path})
        mounts = [
            m
            for m in worker["volumeMounts"]
            if m.get("mountPath") == "/sgl-workspace/sglang-fix"
        ]
        if len(mounts) != 1 or mounts[0].get("subPath") or mounts[0].get("subPathExpr"):
            raise ValueError(
                f"Expected one complete patch-directory mount on {role_name}"
            )
        volumes = [
            (i, v)
            for i, v in enumerate(template["spec"].get("volumes", []))
            if v["name"] == mounts[0]["name"] and "configMap" in v
        ]
        if len(volumes) != 1:
            raise ValueError(
                f"Expected one ConfigMap backing the patch mount on {role_name}"
            )
        volume_index, volume = volumes[0]
        if volume["configMap"].get("items") or volume["configMap"].get("optional"):
            raise ValueError("The patch ConfigMap must mount all required files")
        volume_path = pointer + f"/spec/volumes/{volume_index}"
        volume_guard = {
            "op": "test",
            "path": volume_path + "/name",
            "value": volume["name"],
        }
        forward.append(volume_guard)
        rollback.append(copy.deepcopy(volume_guard))
        old_image, old_name = worker["image"], volume["configMap"]["name"]
        new_projection = dict(volume["configMap"], name=new_name)
        forward.append(
            {
                "op": "test",
                "path": volume_path + "/configMap",
                "value": copy.deepcopy(volume["configMap"]),
            }
        )
        rollback.append(
            {"op": "test", "path": volume_path + "/configMap", "value": new_projection}
        )
        for path, before, after in (
            (worker_path + "/image", old_image, image),
            (volume_path + "/configMap/name", old_name, new_name),
        ):
            forward.extend(
                [
                    {"op": "test", "path": path, "value": before},
                    {"op": "replace", "path": path, "value": after},
                ]
            )
            rollback.extend(
                [
                    {"op": "test", "path": path, "value": after},
                    {"op": "replace", "path": path, "value": before},
                ]
            )
        candidate_template = helpers._role(candidate, role_name)[2]
        candidate_template["spec"]["containers"][worker_index]["image"] = image
        candidate_template["spec"]["containers"][worker_index]["env"] = new_env
        candidate_template["spec"]["volumes"][volume_index]["configMap"]["name"] = (
            new_name
        )
        old_values[role_name] = {"image": old_image, "configmap": old_name}
    return forward, rollback, candidate, old_values


def build_bundle(
    source_dir,
    source_ref,
    output_dir,
    *,
    deployment=None,
    image=None,
    namespace="default",
):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError(
            "Choose a new output directory; existing bundles are never overwritten"
        )
    if bool(deployment) != bool(image):
        raise ValueError("--deployment and --image must be supplied together")
    if image is not None and (not image.strip() or any(c.isspace() for c in image)):
        raise ValueError("--image must be an explicit image reference")
    rbg, documents = None, []
    if deployment:
        documents = list(
            yaml.load_all(Path(deployment).read_text(), Loader=helpers.UniqueKeyLoader)
        )
        rbgs = [
            d
            for d in documents
            if isinstance(d, dict) and d.get("kind") == "RoleBasedGroup"
        ]
        if len(rbgs) != 1:
            raise ValueError("Expected one RoleBasedGroup input")
        rbg = rbgs[0]
        namespace = rbg["metadata"]["namespace"]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=".v0522-bundle-"
    ) as tmp:
        staging = Path(tmp)
        manifest = export_source(source_dir, source_ref, staging)
        files = {p.name: p.read_text() for p in (staging / "patches").iterdir()}
        data_bytes = sum(len(k.encode()) + len(v.encode()) for k, v in files.items())
        if data_bytes > 1_048_576:
            raise ValueError("Patch ConfigMap exceeds the Kubernetes data size limit")
        content_hash = digest(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        )
        cm_name = "sglang-v0522-glm53-patches-" + content_hash[:12]
        configmap = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": cm_name, "namespace": namespace},
            "immutable": True,
            "data": files,
        }
        (staging / "configmap.yaml").write_text(helpers.yaml_text(configmap))
        review = dict(
            manifest,
            namespace=namespace,
            configmap=cm_name,
            configmap_data_bytes=data_bytes,
            requested_worker_image=image,
            services_modified=False,
            router_modified=False,
        )
        if rbg is not None:
            forward, rollback, candidate, old_values = deployment_patches(
                rbg, cm_name, image
            )
            for name, value in (
                ("runtime-upgrade.patch.json", forward),
                ("runtime-upgrade.rollback.json", rollback),
            ):
                (staging / name).write_text(json.dumps(value, indent=2) + "\n")
            (staging / "runtime-upgrade.rbg.yaml").write_text(
                helpers.yaml_text(candidate)
            )
            (staging / "baseline.rbg.yaml").write_text(helpers.yaml_text(rbg))
            review.update(
                rbg=rbg["metadata"]["name"],
                baseline_workers=old_values,
                input_deployment_sha256=digest(Path(deployment).read_bytes()),
                services_in_input=sum(
                    isinstance(d, dict) and d.get("kind") == "Service"
                    for d in documents
                ),
            )
        (staging / "review.json").write_text(json.dumps(review, indent=2) + "\n")
        for name in (
            "README.md",
            "MIGRATION.md",
            "VALIDATION.md",
            "legacy-patches.json",
        ):
            path = PACKAGE / name
            if path.exists():
                shutil.copyfile(path, staging / name)
        checksums = [
            f"{digest(p.read_bytes())}  {p.relative_to(staging)}\n"
            for p in sorted(staging.rglob("*"))
            if p.is_file()
        ]
        (staging / "SHA256SUMS").write_text("".join(checksums))
        staging.rename(output_dir)
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=PACKAGE.parents[3])
    parser.add_argument("--source-ref", default="HEAD")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deployment", type=Path)
    parser.add_argument("--image")
    parser.add_argument("--namespace", default="default")
    args = parser.parse_args()
    review = build_bundle(
        args.source_dir,
        args.source_ref,
        args.output_dir,
        deployment=args.deployment,
        image=args.image,
        namespace=args.namespace,
    )
    print(json.dumps(review, indent=2))


if __name__ == "__main__":
    main()
