# SPDX-License-Identifier: Apache-2.0
"""Exercise release exports, the real installer and guarded upgrade/rollback."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "examples/runtime/glm53_pd/release_v0522"
spec = importlib.util.spec_from_file_location(
    "glm53_v0522_bundle", PACKAGE / "build_bundle.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def apply_json_patch(document, operations):
    result = copy.deepcopy(document)
    for operation in operations:
        parts = operation["path"].strip("/").split("/")
        node = result
        for part in parts[:-1]:
            node = node[int(part)] if isinstance(node, list) else node[part]
        key = int(parts[-1]) if isinstance(node, list) else parts[-1]
        if operation["op"] == "test":
            if node[key] != operation["value"]:
                raise ValueError("JSON Patch precondition failed")
        elif operation["op"] in ("replace", "add"):
            node[key] = copy.deepcopy(operation["value"])
        else:
            assert operation["op"] == "remove"
            del node[key]
    return result


def deployment():
    roles = []
    for role in ("prefill", "decode"):
        roles.append(
            {
                "name": role,
                "replicas": 1,
                "standalonePattern": {
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "worker",
                                    "image": f"example/sglang:{role}-old",
                                    "command": [
                                        "bash",
                                        "-c",
                                        "set -euo pipefail\nbash /sgl-workspace/sglang-fix/fix-patch.sh\nexec sglang serve --model-path MODEL_PATH\n",
                                    ],
                                    "volumeMounts": [
                                        {
                                            "name": "patch",
                                            "mountPath": "/sgl-workspace/sglang-fix",
                                            "readOnly": True,
                                        }
                                    ],
                                }
                            ],
                            "volumes": [
                                {
                                    "name": "patch",
                                    "configMap": {"name": role + "-patch-old"},
                                }
                            ],
                        }
                    }
                },
            }
        )
    roles.extend(
        [{"name": "router", "replicas": 1}, {"name": "mooncake", "replicas": 1}]
    )
    return {
        "apiVersion": "workloads.x-k8s.io/v1alpha1",
        "kind": "RoleBasedGroup",
        "metadata": {"name": "test-pd", "namespace": "test"},
        "spec": {"roles": roles},
    }


class TestReleaseBundle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.invalid")
        self.write("python/sglang/__init__.py", "")
        self.write("python/sglang/version.py", '__version__ = "0.5.22"\n')
        self.write("python/sglang/example.py", "VALUE = 1\n")
        self.write("sgl-model-gateway/src/example.rs", "const VALUE: u8 = 1;\n")
        self.git("add", ".")
        self.git("commit", "-qm", "release")
        self.base = self.git("rev-parse", "HEAD").decode().strip()
        self.write("python/sglang/example.py", "VALUE = 2\n")
        self.write("python/sglang/added.py", "ADDED = True\n")
        self.write("sgl-model-gateway/src/example.rs", "const VALUE: u8 = 2;\n")
        self.git("add", ".")
        self.git("commit", "-qm", "migration")
        self.head = self.git("rev-parse", "HEAD").decode().strip()

    def git(self, *args):
        return subprocess.check_output(
            ["git", *args], cwd=self.repo, stderr=subprocess.PIPE
        )

    def write(self, path, content):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def build(self, **kwargs):
        output = self.root / "bundle"
        original = builder.export_source

        def export(*args):
            return original(*args, base_revision=self.base)

        with patch.object(builder, "export_source", side_effect=export):
            review = builder.build_bundle(self.repo, "HEAD", output, **kwargs)
        return output, review

    def pristine_source(self):
        target = self.root / "source"
        for path in ("__init__.py", "version.py", "example.py"):
            dest = target / "python/sglang" / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(self.git("show", f"{self.base}:python/sglang/{path}"))
        return target

    def install(self, output, source, *extra):
        return subprocess.run(
            [
                sys.executable,
                str(output / "patches/apply-stack.py"),
                "--source-dir",
                str(source),
                "--patch-dir",
                str(output / "patches"),
                *extra,
            ],
            text=True,
            capture_output=True,
        )

    def test_real_export_replay_and_idempotence(self):
        output, review = self.build(namespace="test")
        stack = json.loads((output / "patches/stack.json").read_text())
        self.assertEqual(stack["states"]["upstream"], stack["states"]["baseline"])
        self.assertEqual(stack["source_revision"], self.head)
        self.assertIn("python/sglang/version.py", stack["states"]["candidate"])
        source = self.pristine_source()
        dry_run = self.install(output, source, "--dry-run")
        self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
        self.assertEqual(
            (source / "python/sglang/example.py").read_text(), "VALUE = 1\n"
        )
        result = self.install(output, source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            (source / "python/sglang/example.py").read_text(), "VALUE = 2\n"
        )
        self.assertEqual(
            (source / "python/sglang/added.py").read_text(), "ADDED = True\n"
        )
        repeated = self.install(output, source)
        self.assertEqual(repeated.returncode, 0, repeated.stderr)
        self.assertIn('"state": "candidate"', repeated.stdout)
        self.assertTrue(review["gateway_requires_rebuild"])
        self.assertFalse(review["deployed"])
        self.assertNotIn(
            "sgl-model-gateway",
            (output / "patches/0001-glm53-pd-runtime.patch").read_text(),
        )
        self.assertIn("const VALUE: u8 = 2", (output / "gateway.patch").read_text())
        cm = yaml.safe_load((output / "configmap.yaml").read_text())
        self.assertTrue(cm["immutable"])
        self.assertEqual(
            cm["data"], {p.name: p.read_text() for p in (output / "patches").iterdir()}
        )

    def test_wrong_release_refuses_before_any_source_mutation(self):
        output, _ = self.build()
        source = self.pristine_source()
        (source / "python/sglang/version.py").write_text('__version__ = "0.5.18"\n')
        before = {str(p): p.read_bytes() for p in source.rglob("*.py")}
        result = self.install(output, source)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(before, {str(p): p.read_bytes() for p in source.rglob("*.py")})

    def test_upgrade_and_rollback_change_only_images_patches_and_tree_profile(self):
        rbg = deployment()
        path = self.root / "deployment.yaml"
        path.write_text(
            yaml.safe_dump_all([rbg, {"kind": "Service", "spec": {"type": "NodePort"}}])
        )
        output, review = self.build(deployment=path, image="example/sglang:v0.5.22")
        forward = json.loads((output / "runtime-upgrade.patch.json").read_text())
        reverse = json.loads((output / "runtime-upgrade.rollback.json").read_text())
        candidate = apply_json_patch(rbg, forward)
        self.assertEqual(apply_json_patch(candidate, reverse), rbg)
        self.assertEqual(
            candidate, yaml.safe_load((output / "runtime-upgrade.rbg.yaml").read_text())
        )
        self.assertEqual(len([op for op in forward if op["op"] == "replace"]), 4)
        self.assertEqual(len([op for op in forward if op["op"] == "add"]), 2)
        self.assertEqual(
            review["required_worker_env"], {builder.TREE_CORE_ENV: "python"}
        )
        for role in ("prefill", "decode"):
            worker = builder.helpers._role(candidate, role)[2]["spec"]["containers"][0]
            self.assertEqual(
                worker["env"], [{"name": builder.TREE_CORE_ENV, "value": "python"}]
            )
        self.assertEqual(candidate["spec"]["roles"][2:], rbg["spec"]["roles"][2:])
        self.assertEqual(review["services_in_input"], 1)
        self.assertFalse(review["services_modified"])
        altered = copy.deepcopy(rbg)
        altered["spec"]["roles"][0]["standalonePattern"]["template"]["spec"][
            "containers"
        ][0]["image"] = "changed"
        with self.assertRaises(ValueError):
            apply_json_patch(altered, forward)

        altered = copy.deepcopy(rbg)
        altered["spec"]["roles"][0]["standalonePattern"]["template"]["spec"]["volumes"][
            0
        ]["configMap"]["items"] = [{"key": "partial", "path": "partial"}]
        with self.assertRaises(ValueError):
            apply_json_patch(altered, forward)

    def test_existing_environment_is_preserved_guarded_and_reversible(self):
        rbg = deployment()
        for role in ("prefill", "decode"):
            worker = builder.helpers._role(rbg, role)[2]["spec"]["containers"][0]
            worker["env"] = [
                {
                    "name": "OTHER",
                    "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}},
                },
                {
                    "name": builder.TREE_CORE_ENV,
                    "valueFrom": {"configMapKeyRef": {"name": "old", "key": "backend"}},
                },
            ]
        forward, reverse, expected, _ = builder.deployment_patches(
            rbg, "new-cm", "v0.5.22"
        )
        actual = apply_json_patch(rbg, forward)
        self.assertEqual(actual, expected)
        self.assertEqual(apply_json_patch(actual, reverse), rbg)
        self.assertEqual(len([op for op in forward if op["op"] == "replace"]), 6)
        for role in ("prefill", "decode"):
            before = builder.helpers._role(rbg, role)[2]["spec"]["containers"][0]["env"]
            after = builder.helpers._role(actual, role)[2]["spec"]["containers"][0][
                "env"
            ]
            self.assertEqual(after[0], before[0])
            self.assertEqual(
                after[1], {"name": builder.TREE_CORE_ENV, "value": "python"}
            )
        altered = copy.deepcopy(rbg)
        builder.helpers._role(altered, "prefill")[2]["spec"]["containers"][0][
            "env"
        ].append({"name": "DRIFT", "value": "1"})
        with self.assertRaises(ValueError):
            apply_json_patch(altered, forward)

    def test_duplicate_tree_environment_is_rejected(self):
        rbg = deployment()
        worker = builder.helpers._role(rbg, "prefill")[2]["spec"]["containers"][0]
        worker["env"] = [{"name": builder.TREE_CORE_ENV, "value": "python"}] * 2
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            builder.deployment_patches(rbg, "new-cm", "v0.5.22")

    def test_uncommitted_runtime_or_gateway_is_not_silently_omitted(self):
        for path in (
            "python/sglang/example.py",
            "sgl-model-gateway/src/example.rs",
            "python/sglang/untracked.py",
        ):
            with self.subTest(path=path):
                target = self.repo / path
                before = target.read_bytes() if target.exists() else None
                target.write_text("uncommitted\n")
                with self.assertRaisesRegex(ValueError, "Commit all"):
                    self.build()
                if before is None:
                    target.unlink()
                else:
                    target.write_bytes(before)

    def test_deployment_requires_explicit_image(self):
        with self.assertRaisesRegex(ValueError, "together"):
            self.build(deployment=self.root / "missing.yaml")

    def test_partial_or_optional_patch_mount_is_rejected(self):
        for field, value in (
            ("items", [{"key": "one", "path": "one"}]),
            ("optional", True),
        ):
            with self.subTest(field=field):
                rbg = deployment()
                rbg["spec"]["roles"][0]["standalonePattern"]["template"]["spec"][
                    "volumes"
                ][0]["configMap"][field] = value
                with self.assertRaisesRegex(ValueError, "all required"):
                    builder.deployment_patches(rbg, "new-cm", "image:v0.5.22")

    def test_real_startup_stops_if_python_imports_a_different_checkout(self):
        output, _ = self.build()
        source = self.pristine_source()
        wrong = self.root / "wrong"
        (wrong / "sglang").mkdir(parents=True)
        (wrong / "sglang/__init__.py").write_text("")
        served = self.root / "served"
        env = dict(
            os.environ,
            PYTHONPATH=str(wrong),
            SGLANG_PATCH_SOURCE_DIR=str(source),
            SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND="python",
        )
        result = subprocess.run(
            [
                "bash",
                "-c",
                'set -euo pipefail\nbash "$1"\ntouch "$2"',
                "startup",
                str(output / "patches/fix-patch.sh"),
                str(served),
            ],
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(served.exists())
        self.assertEqual(
            (source / "python/sglang/example.py").read_text(), "VALUE = 1\n"
        )
        env["PYTHONPATH"] = str(source / "python")
        for unsupported in (None, "rust"):
            if unsupported is None:
                env.pop(builder.TREE_CORE_ENV, None)
            else:
                env[builder.TREE_CORE_ENV] = unsupported
            refused = subprocess.run(
                ["bash", str(output / "patches/fix-patch.sh")],
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("requires", refused.stderr)
            self.assertEqual(
                (source / "python/sglang/example.py").read_text(), "VALUE = 1\n"
            )
        env[builder.TREE_CORE_ENV] = "python"
        result = subprocess.run(
            [
                "bash",
                "-c",
                'set -euo pipefail\nbash "$1"\ntouch "$2"',
                "startup",
                str(output / "patches/fix-patch.sh"),
                str(served),
            ],
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(served.exists())


if __name__ == "__main__":
    unittest.main()
