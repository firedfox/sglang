"""CPU checks for verified, reversible pd1 rollout bundles.

Fixtures are synthetic. No private deployment, model, cluster or network is
required; the generated shell entrypoint runs a harmless local fake installer.
"""

import copy
import hashlib
import importlib.util
import json
import runpy
import shlex
import stat
import subprocess
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "examples/runtime/glm53_pd/backport_v0518/build_bundle.py"
runpy.run_path(str(REPO_ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
](est_time=3, suite="base-a-test-cpu")
spec = importlib.util.spec_from_file_location("glm53_pd_bundle", SCRIPT)
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def apply_json_patch(original, operations):
    """Apply the emitted RFC 6902 test/replace operations independently."""
    result = copy.deepcopy(original)
    for operation in operations:
        parts = [
            part.replace("~1", "/").replace("~0", "~")
            for part in operation["path"].split("/")[1:]
        ]
        parent = result
        try:
            for part in parts[:-1]:
                parent = parent[int(part)] if isinstance(parent, list) else parent[part]
            key = int(parts[-1]) if isinstance(parent, list) else parts[-1]
            current = parent[key]
        except (IndexError, KeyError, ValueError, TypeError) as error:
            raise ValueError("JSON Patch path does not exist") from error
        if operation["op"] == "test":
            if current != operation["value"]:
                raise ValueError("JSON Patch precondition failed")
        elif operation["op"] == "replace":
            parent[key] = copy.deepcopy(operation["value"])
        else:
            raise AssertionError("Unexpected emitted JSON Patch operation")
    return result


class TestGLM53PDBundle(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / "package"
        (self.package / "patches").mkdir(parents=True)
        self.archive = self.root / "patches.zip"
        self.configmap = self.root / "live-configmap.yaml"
        self.deployment = self.root / "deployment.yaml"
        self.output = self.root / "bundle"
        self.baseline_files = {
            "0001-base.patch": "--- a/python/sglang/sample.py\n+++ b/python/sglang/sample.py\n@@ -1 +1 @@\n-old\n+base\n",
            "0002-next.patch": "--- a/python/sglang/sample.py\n+++ b/python/sglang/sample.py\n@@ -1 +1 @@\n-base\n+next\n",
            "fix-patch.sh": "#!/bin/bash\nexit 0\n",
            "pre-handle.sh": "#!/bin/bash\necho baseline\n",
        }
        self.extension_name = "0042-extra.patch"
        self.extension = "--- a/python/sglang/sample.py\n+++ b/python/sglang/sample.py\n@@ -1 +1 @@\n-next\n+candidate\n"
        (self.package / "patches" / self.extension_name).write_text(self.extension)
        self.installer = "import json, sys\nprint(json.dumps(sys.argv[1:]))\n"
        (self.package / "apply_stack.py").write_text(self.installer)
        self.old_name = "sglang-fix-patch"
        self.cm = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": self.old_name,
                "namespace": "unit-ns",
                "resourceVersion": "123",
                "uid": "old-uid",
            },
            "data": dict(self.baseline_files),
        }
        self.rbg = {
            "apiVersion": "workloads.x-k8s.io/v1alpha1",
            "kind": "RoleBasedGroup",
            "metadata": {"name": "pd1-unit", "namespace": "unit-ns"},
            "spec": {
                "roles": [
                    self.role("mooncake-master", 2),
                    self.role("router"),
                    self.role("prefill"),
                    self.role("decode"),
                    self.role("mooncake-store"),
                    self.role("standby", 0),
                ],
                "roleTemplates": [
                    {
                        "name": "unrelated",
                        "template": {"spec": {"priorityClassName": "preserved"}},
                    }
                ],
            },
        }
        self.service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "pd1-unit", "namespace": "unit-ns"},
            "spec": {
                "type": "NodePort",
                "ports": [{"port": 8000, "targetPort": 8000, "nodePort": 31180}],
                "selector": {"role": "router"},
            },
        }
        router = self.role_template(self.rbg, "router")["spec"]["containers"][0]
        router["name"] = "router-container"
        router["command"] = [
            "bash",
            "-c",
            "exec python3 -m sglang_router.launch_router \\\n  --pd-disaggregation \\\n  --dp-aware \\\n  --port 8000\n",
        ]
        router["env"] = [
            {"name": "KEEP", "value": "unchanged"},
            {"name": "RUST_LOG", "value": "debug,sgl_model_gateway=debug"},
        ]
        self.write_baseline()
        self.stack = {
            "format_version": 1,
            "base_revision": "1" * 40,
            "patches": [
                *[
                    {"name": name, "sha256": digest(value), "stage": "baseline"}
                    for name, value in self.baseline_files.items()
                    if name.endswith(".patch")
                ],
                {
                    "name": self.extension_name,
                    "sha256": digest(self.extension),
                    "stage": "extension",
                },
            ],
            "states": {
                name: {"python/sglang/sample.py": digest(name)}
                for name in ("upstream", "baseline", "candidate")
            },
        }
        self.write_stack()
        self.write_deployment()

    def role(self, name, replicas=1):
        return {
            "name": name,
            "replicas": replicas,
            "standalonePattern": {
                "template": {
                    "metadata": {"labels": {"role": name}},
                    "spec": {
                        "containers": [
                            {
                                "name": "worker",
                                "image": "example.invalid/sglang:v0.5.18",
                                "command": [
                                    "bash",
                                    "-c",
                                    "set -euo pipefail\nbash /sgl-workspace/sglang-fix/fix-patch.sh\nexec sglang serve --model-path /models/test\n",
                                ],
                                "env": [{"name": "PRESERVE", "value": "literal"}],
                                "volumeMounts": [
                                    {
                                        "name": "runtime-patches",
                                        "mountPath": "/sgl-workspace/sglang-fix",
                                        "readOnly": True,
                                    }
                                ],
                            }
                        ],
                        "volumes": [
                            {"name": "scratch", "emptyDir": {}},
                            {
                                "name": "runtime-patches",
                                "configMap": {
                                    "name": self.old_name,
                                    "defaultMode": 493,
                                },
                            },
                        ],
                    },
                }
            },
        }

    @staticmethod
    def role_template(rbg, name):
        return next(role for role in rbg["spec"]["roles"] if role["name"] == name)[
            "standalonePattern"
        ]["template"]

    def write_archive(self, members=None):
        members = (
            members
            if members is not None
            else [
                ("patches/" + name, value)
                for name, value in self.baseline_files.items()
            ]
        )
        with zipfile.ZipFile(self.archive, "w") as archive:
            for name, value in members:
                archive.writestr(name, value)

    def write_baseline(self):
        self.baseline = {
            "upstream_revision": "1" * 40,
            "files": {
                name: digest(value) for name, value in self.baseline_files.items()
            },
        }
        (self.package / "baseline_inputs.json").write_text(json.dumps(self.baseline))
        self.cm["data"] = dict(self.baseline_files)
        self.configmap.write_text(yaml.safe_dump(self.cm, sort_keys=False))
        self.write_archive()

    def write_stack(self):
        (self.package / "stack.json").write_text(json.dumps(self.stack))

    def write_deployment(self):
        self.deployment.write_text(
            yaml.safe_dump_all([self.rbg, self.service], sort_keys=False)
        )

    def build(self, output=None):
        return bundle.build_bundle(
            self.archive,
            self.configmap,
            self.deployment,
            output or self.output,
            package=self.package,
        )

    def read_json(self, name):
        return json.loads((self.output / name).read_text())

    def assert_rejected(self, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            self.build()
        self.assertFalse(self.output.exists())

    def test_archive_and_configmap_require_every_identical_baseline_byte(self):
        contents, configmap = bundle.verified_inputs(
            self.archive, self.configmap, self.package
        )
        self.assertEqual(contents, self.baseline_files)
        self.assertEqual(configmap, self.cm)
        self.cm["data"]["pre-handle.sh"] += "# tampered\n"
        self.configmap.write_text(yaml.safe_dump(self.cm))
        self.assert_rejected("ConfigMap data must match")

    def test_archive_rejects_missing_unknown_duplicate_and_tampered_files(self):
        members = [
            ("patches/" + name, value) for name, value in self.baseline_files.items()
        ]
        cases = (
            (members[:-1], "complete baseline"),
            (members + [("patches/0099-unexpected.patch", "new")], "unexpected"),
            (members + [members[0]], "Duplicate"),
            ([(members[0][0], "tampered")] + members[1:], "hash mismatch"),
        )
        for archive_members, error in cases:
            with self.subTest(error=error), warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.write_archive(archive_members)
                self.assert_rejected(error)

    def test_archive_rejects_traversal_absolute_paths_and_symlinks(self):
        for path in (
            "../escaped",
            "patches/../escaped",
            "/patches/escaped",
            "other/file",
        ):
            with self.subTest(path=path):
                self.write_archive([(path, "not an input")])
                self.assert_rejected("Unsafe|Unexpected")
        symlink = zipfile.ZipInfo("patches/fix-patch.sh")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.write_archive([(symlink, "../../outside")])
        self.assert_rejected("Unsafe")
        self.assertFalse((self.root / "escaped").exists())

    def test_archive_member_and_configmap_utf8_size_limits_are_enforced(self):
        self.baseline_files["pre-handle.sh"] = "a" * 2_000_001
        self.write_baseline()
        self.assert_rejected("Oversized archive member")
        self.baseline_files["pre-handle.sh"] = "字" * 400_000
        self.write_baseline()
        self.assert_rejected("ConfigMap exceeds")

    def test_duplicate_yaml_keys_and_binary_data_are_rejected(self):
        original = self.configmap.read_text()
        self.configmap.write_text(original + "data: {}\n")
        self.assert_rejected("Duplicate YAML key")
        self.configmap.write_text(original)
        self.cm["binaryData"] = {"extra": "eA=="}
        self.configmap.write_text(yaml.safe_dump(self.cm))
        self.assert_rejected("ConfigMap data must match")
        self.configmap.write_text(original)
        self.deployment.write_text(
            self.deployment.read_text().replace(
                "kind: RoleBasedGroup", "kind: RoleBasedGroup\nkind: RoleBasedGroup", 1
            )
        )
        self.assert_rejected("Duplicate YAML key")

    def test_stack_rejects_unsafe_duplicate_missing_or_reordered_patches(self):
        original = copy.deepcopy(self.stack)
        cases = []
        for name in (
            "../escaped.patch",
            "/escaped.patch",
            "0042-../escaped.patch",
            "fix-patch.sh",
        ):
            altered = copy.deepcopy(original)
            altered["patches"][-1]["name"] = name
            cases.append(altered)
        duplicate = copy.deepcopy(original)
        duplicate["patches"].append(copy.deepcopy(duplicate["patches"][-1]))
        cases.append(duplicate)
        missing = copy.deepcopy(original)
        missing["patches"].pop(0)
        cases.append(missing)
        reordered = copy.deepcopy(original)
        reordered["patches"][0:2] = reversed(reordered["patches"][0:2])
        cases.append(reordered)
        collision = copy.deepcopy(original)
        collision["patches"][-1]["name"] = collision["patches"][0]["name"]
        cases.append(collision)
        for i, stack in enumerate(cases):
            with self.subTest(case=i):
                self.stack = stack
                self.write_stack()
                self.assert_rejected("safe numbered|complete ordered")

    def test_stack_rejects_wrong_version_revision_digest_stage_or_source_states(self):
        original = copy.deepcopy(self.stack)
        cases = []
        for key, value in (("format_version", 2), ("base_revision", "2" * 40)):
            altered = copy.deepcopy(original)
            altered[key] = value
            cases.append(altered)
        for key, value in (("sha256", "not-a-digest"), ("stage", "other")):
            altered = copy.deepcopy(original)
            altered["patches"][-1][key] = value
            cases.append(altered)
        missing = copy.deepcopy(original)
        missing["states"].pop("baseline")
        cases.append(missing)
        mismatched = copy.deepcopy(original)
        mismatched["states"]["candidate"] = {
            "python/sglang/other.py": digest("different")
        }
        cases.append(mismatched)
        escaped = copy.deepcopy(original)
        escaped["states"] = {
            name: {"python/sglang/../../outside": digest(name)}
            for name in original["states"]
        }
        cases.append(escaped)
        for i, stack in enumerate(cases):
            with self.subTest(case=i):
                self.stack = stack
                self.write_stack()
                self.assert_rejected(
                    "Unsupported|Invalid patch|all three|same affected|inside python/sglang"
                )

    def test_changed_extension_content_cannot_build_a_candidate(self):
        (self.package / "patches" / self.extension_name).write_text(
            self.extension + "# changed\n"
        )
        self.assert_rejected("Backport hash mismatch")

    def test_runtime_patch_changes_only_two_worker_volume_references_and_rolls_back(
        self,
    ):
        input_bytes = self.deployment.read_bytes()
        review = self.build()
        change = self.read_json("runtime-only.patch.json")
        revert = self.read_json("runtime-only.rollback.json")
        expected = copy.deepcopy(self.rbg)
        for role in ("prefill", "decode"):
            self.role_template(expected, role)["spec"]["volumes"][1]["configMap"][
                "name"
            ] = review["candidate_configmap"]
        applied = apply_json_patch(self.rbg, change)
        self.assertEqual(applied, expected)
        self.assertEqual(apply_json_patch(applied, revert), self.rbg)
        self.assertEqual(
            yaml.safe_load((self.output / "runtime-only.rbg.yaml").read_text()),
            expected,
        )
        self.assertEqual(
            yaml.safe_load((self.output / "baseline.rbg.yaml").read_text()), self.rbg
        )
        self.assertEqual(self.deployment.read_bytes(), input_bytes)
        self.assertEqual(review["service_documents_in_input"], 1)
        self.assertFalse(review["services_modified"])
        self.assertFalse(review["deployed"])
        self.assertFalse(review["prefill_decode_parameters_modified"])
        self.assertEqual(len([op for op in change if op["op"] == "replace"]), 2)
        self.assertEqual(
            next(r for r in applied["spec"]["roles"] if r["name"] == "standby")[
                "replicas"
            ],
            0,
        )

    def test_json_patch_guards_reject_wrong_identity_order_and_newer_config(self):
        self.build()
        change = self.read_json("runtime-only.patch.json")
        wrong_name = copy.deepcopy(self.rbg)
        wrong_name["metadata"]["name"] = "other"
        wrong_namespace = copy.deepcopy(self.rbg)
        wrong_namespace["metadata"]["namespace"] = "other"
        reordered = copy.deepcopy(self.rbg)
        reordered["spec"]["roles"][2:4] = reversed(reordered["spec"]["roles"][2:4])
        newer_config = copy.deepcopy(self.rbg)
        self.role_template(newer_config, "decode")["spec"]["volumes"][1]["configMap"][
            "name"
        ] = "someone-elses-change"
        changed_startup = copy.deepcopy(self.rbg)
        command = self.role_template(changed_startup, "decode")["spec"]["containers"][
            0
        ]["command"]
        command[2] = command[2].replace("set -euo pipefail\n", "")
        renamed_worker = copy.deepcopy(self.rbg)
        self.role_template(renamed_worker, "prefill")["spec"]["containers"][0][
            "name"
        ] = "different-worker"
        for document in (
            wrong_name,
            wrong_namespace,
            reordered,
            newer_config,
            changed_startup,
            renamed_worker,
        ):
            with self.subTest(document=document["metadata"]):
                unchanged = copy.deepcopy(document)
                with self.assertRaisesRegex(ValueError, "precondition"):
                    apply_json_patch(document, change)
                self.assertEqual(document, unchanged)
        with self.assertRaisesRegex(ValueError, "precondition"):
            apply_json_patch(self.rbg, self.read_json("runtime-only.rollback.json"))
        changed_candidate = apply_json_patch(self.rbg, change)
        command = self.role_template(changed_candidate, "decode")["spec"]["containers"][
            0
        ]["command"]
        command[2] = command[2].replace("set -euo pipefail\n", "")
        with self.assertRaisesRegex(ValueError, "precondition"):
            apply_json_patch(
                changed_candidate, self.read_json("runtime-only.rollback.json")
            )

    def test_router_rank_and_log_candidates_are_independent_and_reversible(self):
        self.build()
        rank = self.read_json("router-rank-selection.patch.json")
        logs = self.read_json("router-log-level.patch.json")
        rank_applied = apply_json_patch(self.rbg, rank)
        log_applied = apply_json_patch(self.rbg, logs)
        before = self.role_template(self.rbg, "router")["spec"]["containers"][0]
        after_rank = self.role_template(rank_applied, "router")["spec"]["containers"][0]
        after_log = self.role_template(log_applied, "router")["spec"]["containers"][0]
        self.assertEqual(after_rank["env"], before["env"])
        old_tokens = shlex.split(before["command"][2].replace("\\\n", " "))
        new_tokens = shlex.split(after_rank["command"][2].replace("\\\n", " "))
        self.assertEqual(
            new_tokens, [token for token in old_tokens if token != "--dp-aware"]
        )
        self.assertEqual(after_log["command"], before["command"])
        self.assertEqual(after_log["env"][1], {"name": "RUST_LOG", "value": "info"})
        self.assertEqual(
            apply_json_patch(
                rank_applied, self.read_json("router-rank-selection.rollback.json")
            ),
            self.rbg,
        )
        self.assertEqual(
            apply_json_patch(
                log_applied, self.read_json("router-log-level.rollback.json")
            ),
            self.rbg,
        )
        self.assertEqual(
            apply_json_patch(rank_applied, logs), apply_json_patch(log_applied, rank)
        )
        for role in (
            "prefill",
            "decode",
            "mooncake-master",
            "mooncake-store",
            "standby",
        ):
            self.assertEqual(
                self.role_template(rank_applied, role),
                self.role_template(self.rbg, role),
            )
            self.assertEqual(
                self.role_template(log_applied, role),
                self.role_template(self.rbg, role),
            )

    def test_candidate_configmap_is_immutable_versioned_and_matches_patch_files(self):
        review = self.build()
        configmap = yaml.safe_load((self.output / "configmap.yaml").read_text())
        self.assertTrue(configmap["immutable"])
        self.assertRegex(
            review["candidate_configmap"], r"^sglang-fix-patch-[0-9a-f]{12}$"
        )
        self.assertEqual(
            configmap["metadata"],
            {"name": review["candidate_configmap"], "namespace": "unit-ns"},
        )
        self.assertEqual(review["patch_count"], 3)
        self.assertEqual(
            review["configmap_data_bytes"],
            sum(
                len(k.encode()) + len(v.encode()) for k, v in configmap["data"].items()
            ),
        )
        for name, value in configmap["data"].items():
            self.assertEqual((self.output / "patches" / name).read_text(), value)
        for name, value in self.baseline_files.items():
            if name != "fix-patch.sh":
                self.assertEqual(configmap["data"][name], value)
        self.assertEqual(configmap["data"]["apply-stack.py"], self.installer)
        self.assertEqual(json.loads(configmap["data"]["stack.json"]), self.stack)
        again = self.build(self.root / "same-inputs")
        self.assertEqual(again["candidate_configmap"], review["candidate_configmap"])
        (self.package / "apply_stack.py").write_text(
            self.installer + "# different build\n"
        )
        changed = self.build(self.root / "different-installer")
        self.assertNotEqual(
            changed["candidate_configmap"], review["candidate_configmap"]
        )

    def test_shell_entrypoint_uses_sibling_installer_with_literal_patch_directory(self):
        output = self.root / "bundle with $(must-not-run)"
        self.build(output)
        command = subprocess.run(
            ["bash", str(output / "patches" / "fix-patch.sh")],
            capture_output=True,
            text=True,
            check=True,
            cwd=self.root,
        )
        self.assertEqual(
            json.loads(command.stdout),
            [
                "--source-dir",
                "/sgl-workspace/sglang",
                "--patch-dir",
                str(output / "patches"),
            ],
        )

    def test_existing_output_is_not_overwritten(self):
        self.output.mkdir()
        marker = self.output / "keep"
        marker.write_text("do not change")
        self.assert_rejected_existing_output()
        self.assertEqual(marker.read_text(), "do not change")
        marker.unlink()
        self.assert_rejected_existing_output()
        self.output.rmdir()
        self.output.write_text("existing file")
        self.assert_rejected_existing_output()
        self.assertEqual(self.output.read_text(), "existing file")

    def assert_rejected_existing_output(self):
        with self.assertRaisesRegex(
            ValueError, "existing bundles are never overwritten"
        ):
            self.build()

    def test_role_namespace_and_volume_ambiguities_fail_before_output(self):
        original = copy.deepcopy(self.rbg)
        altered = copy.deepcopy(original)
        altered["metadata"]["namespace"] = "different"
        cases = [(altered, "same namespace")]
        duplicate_role = copy.deepcopy(original)
        duplicate_role["spec"]["roles"].append(
            copy.deepcopy(duplicate_role["spec"]["roles"][2])
        )
        cases.append((duplicate_role, "one prefill role"))
        duplicate_volume = copy.deepcopy(original)
        volumes = self.role_template(duplicate_volume, "decode")["spec"]["volumes"]
        volumes.append(copy.deepcopy(volumes[1]))
        cases.append((duplicate_volume, "one baseline ConfigMap volume"))
        missing_volume = copy.deepcopy(original)
        self.role_template(missing_volume, "prefill")["spec"]["volumes"] = []
        cases.append((missing_volume, "one baseline ConfigMap volume"))
        for document, error in cases:
            with self.subTest(error=error):
                self.rbg = document
                self.write_deployment()
                self.assert_rejected(error)

    def test_worker_startup_rejects_missing_or_disabled_errexit_and_bad_order(self):
        bootstrap = "bash /sgl-workspace/sglang-fix/fix-patch.sh\n"
        serve = "exec sglang serve --model-path /models/test\n"
        scripts = (
            bootstrap + serve,
            "set -e\n" + bootstrap + serve,
            "set -euo pipefail\nset +e\n" + bootstrap + serve,
            "set -euo pipefail\n" + bootstrap + "set +e\n" + serve,
            "set -euo pipefail\n" + serve + bootstrap,
            "set -euo pipefail\n" + bootstrap + bootstrap + serve,
            "set -euo pipefail\nbash /sgl-workspace/sglang-fix/fix-patch.sh || true\n"
            + serve,
            "set -euo pipefail\nif\n" + bootstrap + "then\n  :\nfi\n" + serve,
        )
        for role in ("prefill", "decode"):
            original = copy.deepcopy(self.rbg)
            for script in scripts:
                with self.subTest(role=role, script=script):
                    self.rbg = copy.deepcopy(original)
                    self.role_template(self.rbg, role)["spec"]["containers"][0][
                        "command"
                    ][2] = script
                    self.write_deployment()
                    self.assert_rejected("fail-closed|precede worker startup")
            self.rbg = original

    def test_worker_startup_requires_the_verified_bash_form(self):
        original = copy.deepcopy(self.rbg)
        command = self.role_template(original, "prefill")["spec"]["containers"][0][
            "command"
        ]
        for replacement in (
            ["sh", "-c", command[2]],
            [*command, "unexpected-argument"],
        ):
            with self.subTest(command=replacement):
                self.rbg = copy.deepcopy(original)
                self.role_template(self.rbg, "prefill")["spec"]["containers"][0][
                    "command"
                ] = replacement
                self.write_deployment()
                self.assert_rejected("verified bash startup form")

    def test_worker_startup_accepts_comments_and_post_bootstrap_export_unchanged(self):
        command = self.role_template(self.rbg, "prefill")["spec"]["containers"][0][
            "command"
        ]
        command[2] = (
            "# The verified startup may contain comments.\n\nset -euo pipefail\n"
            "# Install before any worker starts.\n"
            "bash /sgl-workspace/sglang-fix/fix-patch.sh\n"
            "# A previous optional pre-handle call is commented out.\n"
            "export SGLANG_WARMUP_TIMEOUT=3600\n"
            "exec sglang serve --model-path /models/test\n"
        )
        self.write_deployment()
        self.build()
        candidate = yaml.safe_load((self.output / "runtime-only.rbg.yaml").read_text())
        self.assertEqual(
            self.role_template(candidate, "prefill")["spec"]["containers"][0][
                "command"
            ],
            command,
        )

    def test_worker_shell_does_not_serve_after_actual_installer_failure(self):
        (self.package / "apply_stack.py").write_text("raise SystemExit(17)\n")
        self.build()
        marker = self.root / "worker-started"
        worker = self.root / "fake-worker.py"
        worker.write_text(
            "from pathlib import Path\nimport sys\nPath(sys.argv[1]).write_text('started')\n"
        )
        command = self.role_template(self.rbg, "decode")["spec"]["containers"][0][
            "command"
        ][2]
        command = command.replace(
            "bash /sgl-workspace/sglang-fix/fix-patch.sh",
            shlex.join(["bash", str(self.output / "patches" / "fix-patch.sh")]),
        ).replace(
            "exec sglang serve --model-path /models/test",
            "exec " + shlex.join(["python3", str(worker), str(marker)]),
        )
        result = subprocess.run(
            ["bash", "-c", command],
            check=False,
            capture_output=True,
            text=True,
            cwd=self.root,
        )
        self.assertEqual(result.returncode, 17)
        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
