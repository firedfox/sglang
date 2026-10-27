"""CPU-only command and execution-boundary tests for the GLM-5.3 PD launcher."""

import contextlib
import importlib.util
import io
import json
import os
import runpy
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCH_PATH = REPO_ROOT / "examples/runtime/glm53_pd/launch.py"

# Load the stdlib-only CI marker without importing the SGLang runtime package.
register_cpu_ci = runpy.run_path(
    str(REPO_ROOT / "python/sglang/test/ci/ci_register.py")
)["register_cpu_ci"]
register_cpu_ci(est_time=1, suite="base-a-test-cpu")

spec = importlib.util.spec_from_file_location("glm53_pd_launch", LAUNCH_PATH)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class TestGLM53PDLaunch(unittest.TestCase):
    def plan(self, role, *options, environment=None):
        output = io.StringIO()
        environment = environment if environment is not None else {}
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(launch.os, "execvpe") as execute,
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(launch.main([role, *options]), 0)
            execute.assert_not_called()
        return json.loads(output.getvalue())

    def value(self, plan, flag):
        return plan["argv"][plan["argv"].index(flag) + 1]

    def assert_invalid(self, *arguments, environment=None):
        with (
            patch.dict(os.environ, environment or {}, clear=True),
            patch.object(launch.os, "execvpe") as execute,
            contextlib.redirect_stderr(io.StringIO()),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            launch.main(list(arguments))
        self.assertEqual(error.exception.code, 2)
        execute.assert_not_called()

    def test_router_has_strict_admission_and_worker_level_routing(self):
        plan = self.plan("router")
        self.assertEqual(
            plan["argv"][:3], [sys.executable, "-m", "sglang_router.launch_router"]
        )
        for flag, expected in (
            ("--max-concurrent-requests", "256"),
            ("--rate-limit-tokens-per-second", "0"),
            ("--queue-size", "0"),
            ("--retry-max-retries", "2"),
            ("--prefill-policy", "cache_aware"),
            ("--decode-policy", "power_of_two"),
        ):
            self.assertEqual(self.value(plan, flag), expected)
        for flag in (
            "--dp-aware",
            "--model-path",
            "--chat-template",
            "--reasoning-parser",
            "--tool-call-parser",
        ):
            self.assertNotIn(flag, plan["argv"])
        self.assertEqual(plan["env"], {})

    def test_worker_speculation_and_transfer_formats_match(self):
        for role in ("prefill", "decode"):
            with self.subTest(role=role):
                plan = self.plan(role, "--model-path", "/models/glm53")
                self.assertEqual(plan["argv"][:2], ["sglang", "serve"])
                for flag, expected in (
                    ("--speculative-algorithm", "EAGLE"),
                    ("--speculative-num-steps", "3"),
                    ("--speculative-eagle-topk", "1"),
                    ("--speculative-num-draft-tokens", "4"),
                    ("--kv-cache-dtype", "fp8_e4m3"),
                    ("--page-size", "64"),
                    ("--context-length", "1048576"),
                    ("--disaggregation-mode", role),
                    ("--disaggregation-transfer-backend", "mooncake"),
                    ("--schedule-policy", "fcfs"),
                    ("--reasoning-parser", "glm45"),
                    ("--tool-call-parser", "glm47"),
                ):
                    self.assertEqual(self.value(plan, flag), expected)

    def test_measured_eagle_profile_is_explicit_and_matches_both_workers(self):
        for role in ("prefill", "decode"):
            plan = self.plan(
                role, "--model-path", "/models/glm53", "--eagle-profile", "5/1/6"
            )
            self.assertEqual(self.value(plan, "--speculative-num-steps"), "5")
            self.assertEqual(self.value(plan, "--speculative-eagle-topk"), "1")
            self.assertEqual(self.value(plan, "--speculative-num-draft-tokens"), "6")
        self.assert_invalid("router", "--eagle-profile", "5/1/6")
        self.assert_invalid(
            "decode", "--model-path", "/models/glm53", "--eagle-profile", "6/5"
        )

    def test_admission_queue_overrides_require_admission_and_validate_values(self):
        options = ("--model-path", "/models/glm53", "--enable-dp-admission")
        plan = self.plan(
            "decode",
            *options,
            "--dp-admission-max-queued-requests",
            "64",
            "--dp-admission-queue-timeout",
            "2.5",
        )
        self.assertEqual(self.value(plan, "--dp-admission-max-queued-requests"), "64")
        self.assertEqual(self.value(plan, "--dp-admission-queue-timeout"), "2.5")
        for flag, values in (
            ("--dp-admission-max-queued-requests", ("-1", "1.5")),
            ("--dp-admission-queue-timeout", ("0", "-1", "nan", "inf")),
        ):
            for value in values:
                with self.subTest(flag=flag, value=value):
                    self.assert_invalid("decode", *options, flag, value)
        self.assert_invalid(
            "decode",
            "--model-path",
            "/models/glm53",
            "--dp-admission-max-queued-requests",
            "0",
        )
        self.assert_invalid(
            "decode",
            "--model-path",
            "/models/glm53",
            "--dp-admission-queue-timeout",
            "1",
        )

    def test_prefill_uses_cp8_and_automatic_hicache_sizing(self):
        plan = self.plan("prefill", "--model-path", "/models/glm53")
        for flag, expected in (
            ("--attn-cp-size", "8"),
            ("--cp-strategy", "interleave"),
            ("--chunked-prefill-size", "131072"),
            ("--max-prefill-tokens", "131072"),
            ("--mem-fraction-static", "0.8"),
            ("--hicache-host-memory-fraction", "0.8"),
            ("--hicache-storage-prefetch-policy", "timeout"),
        ):
            self.assertEqual(self.value(plan, flag), expected)
        self.assertIn("--enable-hierarchical-cache", plan["argv"])
        self.assertIn("--enable-prefill-cp", plan["argv"])
        self.assertNotIn("--hicache-size", plan["argv"])
        self.assertNotIn("--hicache-ratio", plan["argv"])
        self.assertNotIn("--enable-dp-attention", plan["argv"])
        self.assertEqual(
            json.loads(self.value(plan, "--deepep-config")),
            {"normal_dispatch": {"num_sms": 64}, "normal_combine": {"num_sms": 64}},
        )

    def test_decode_owns_token_load_balancing_and_profiles_kv_capacity(self):
        plan = self.plan("decode", "--model-path", "/models/glm53")
        for flag, expected in (
            ("--tp-size", "8"),
            ("--dp-size", "8"),
            ("--ep-size", "8"),
            ("--load-balance-method", "total_tokens"),
            ("--max-running-requests", "256"),
            ("--mem-fraction-static", "0.85"),
            ("--deepep-mode", "low_latency"),
        ):
            self.assertEqual(self.value(plan, flag), expected)
        self.assertIn("--enable-dp-attention", plan["argv"])
        self.assertIn("--enable-dp-lm-head", plan["argv"])
        self.assertNotIn("--max-total-tokens", plan["argv"])
        self.assertNotIn("--enable-prefill-cp", plan["argv"])
        self.assertNotIn("--enable-hierarchical-cache", plan["argv"])

    def test_model_path_environment_and_cli_precedence(self):
        for role in ("prefill", "decode"):
            with self.subTest(role=role):
                environment = {"GLM53_MODEL_PATH": "/models/from-env"}
                self.assertEqual(
                    self.value(
                        self.plan(role, environment=environment), "--model-path"
                    ),
                    "/models/from-env",
                )
                plan = self.plan(
                    role,
                    "--model-path",
                    "/models/explicit snapshot $(no-shell)",
                    environment=environment,
                )
                self.assertEqual(
                    self.value(plan, "--model-path"),
                    "/models/explicit snapshot $(no-shell)",
                )

    def test_dp_admission_is_an_explicit_decode_only_opt_in(self):
        baseline = self.plan("decode", "--model-path", "/models/glm53")
        self.assertNotIn("--enable-dp-admission", baseline["argv"])
        plan = self.plan(
            "decode", "--model-path", "/models/glm53", "--enable-dp-admission"
        )
        self.assertEqual(plan["argv"].count("--enable-dp-admission"), 1)
        for flag, expected in (
            ("--dp-admission-max-queued-requests", "0"),
            ("--dp-admission-queue-timeout", "1.0"),
            ("--dp-admission-snapshot-timeout", "30.0"),
        ):
            self.assertEqual(self.value(plan, flag), expected)
        self.assert_invalid("router", "--enable-dp-admission")
        self.assert_invalid(
            "prefill", "--model-path", "/models/glm53", "--enable-dp-admission"
        )

    def test_l3_cost_model_is_an_explicit_prefill_only_opt_in(self):
        flag = "--hicache-storage-prefetch-cost-model"
        model = '{"recompute_tokens_per_second":12345,"read_tokens_per_second":54321}'
        plan = self.plan("prefill", "--model-path", "/models/glm53", flag, model)
        self.assertEqual(
            self.value(plan, "--hicache-storage-prefetch-policy"), "cost_aware"
        )
        self.assertEqual(self.value(plan, flag), model)
        self.assertEqual(plan["argv"].count(flag), 1)
        self.assertEqual(plan["argv"].count("--hicache-storage-prefetch-policy"), 1)
        for invalid in ("not-json", "[]", "1", "null"):
            self.assert_invalid(
                "prefill", "--model-path", "/models/glm53", flag, invalid
            )
        self.assert_invalid("decode", "--model-path", "/models/glm53", flag, model)
        self.assert_invalid("router", flag, model)

    def test_prefill_time_budget_requires_a_calibrated_model_pair(self):
        base = ["prefill", "--model-path", "/models/glm53"]
        model = '{"fixed_ms":1.0,"linear_ms_per_token":0.01}'
        baseline = self.plan(*base)
        self.assertNotIn("--prefill-time-budget-ms", baseline["argv"])
        self.assertNotIn("--prefill-time-model", baseline["argv"])
        plan = self.plan(
            *base, "--prefill-time-budget-ms", "50", "--prefill-time-model", model
        )
        self.assertEqual(self.value(plan, "--prefill-time-budget-ms"), "50.0")
        self.assertEqual(self.value(plan, "--prefill-time-model"), model)
        self.assertEqual(self.value(plan, "--chunked-prefill-size"), "131072")
        self.assertEqual(self.value(plan, "--max-prefill-tokens"), "131072")
        self.assert_invalid(*base, "--prefill-time-budget-ms", "50")
        self.assert_invalid(*base, "--prefill-time-model", model)
        for invalid in ("0", "-1", "nan", "inf", "invalid"):
            self.assert_invalid(
                *base,
                "--prefill-time-budget-ms",
                invalid,
                "--prefill-time-model",
                model,
            )
        for role in ("router", "decode"):
            self.assert_invalid(role, "--prefill-time-budget-ms", "50")

    def test_worker_model_path_is_required(self):
        for role in ("prefill", "decode"):
            with self.subTest(role=role):
                self.assert_invalid(role)
                self.assert_invalid(role, "--model-path", "")
                self.assert_invalid(role, environment={"GLM53_MODEL_PATH": " "})

    def test_unknown_role_and_role_specific_options_are_rejected(self):
        self.assert_invalid("unknown")
        self.assert_invalid("router", "--model-path", "/models/glm53")
        self.assert_invalid("router", "--all-cp-ranks-transfer")
        self.assert_invalid("decode", "--namespace", "wrong-role")
        self.assert_invalid("router", "--ru")
        self.assert_invalid("router", "--unknown-flag")

    def test_ports_and_router_budgets_are_validated(self):
        for value in ("0", "-1", "65536", "invalid"):
            with self.subTest(port=value):
                self.assert_invalid("router", "--port", value)
        for value in ("0", "-1", "invalid"):
            with self.subTest(budget=value):
                self.assert_invalid("router", "--max-concurrent-requests", value)
        self.assert_invalid("router", "--group-name", " ")

    def test_deployment_endpoints_and_router_budget_can_be_overridden(self):
        plan = self.plan(
            "router",
            "--host",
            "10.0.0.20",
            "--port",
            "8100",
            "--namespace",
            "glm-serving",
            "--group-name",
            "glm-test",
            "--worker-port",
            "8200",
            "--prometheus-port",
            "9100",
            "--max-concurrent-requests",
            "128",
            "--otlp-traces-endpoint",
            "collector:4317",
            environment={"POD_IP": "10.0.0.99"},
        )
        for flag, expected in (
            ("--host", "10.0.0.20"),
            ("--port", "8100"),
            ("--service-discovery-namespace", "glm-serving"),
            ("--service-discovery-port", "8200"),
            ("--prometheus-port", "9100"),
            ("--max-concurrent-requests", "128"),
            ("--otlp-traces-endpoint", "collector:4317"),
        ):
            self.assertEqual(self.value(plan, flag), expected)
        self.assertEqual(
            plan["argv"].count("rbg.workloads.x-k8s.io/group-name=glm-test"), 2
        )
        self.assertIn("rbg.workloads.x-k8s.io/role-name=prefill", plan["argv"])
        self.assertIn("rbg.workloads.x-k8s.io/role-name=decode", plan["argv"])

    def test_pod_ip_ib_devices_and_worker_template(self):
        for role in ("prefill", "decode"):
            with self.subTest(role=role):
                plan = self.plan(
                    role,
                    "--chat-template",
                    "/templates/custom template.jinja",
                    "--otlp-traces-endpoint",
                    "collector:4318",
                    environment={
                        "GLM53_MODEL_PATH": "/models/glm53",
                        "POD_IP": "10.0.0.11",
                        "GLM53_IB_DEVICES": "rdma0,rdma1",
                    },
                )
                self.assertEqual(self.value(plan, "--host"), "10.0.0.11")
                self.assertEqual(
                    self.value(plan, "--disaggregation-ib-device"), "rdma0,rdma1"
                )
                self.assertEqual(
                    self.value(plan, "--chat-template"),
                    "/templates/custom template.jinja",
                )
                self.assertEqual(
                    self.value(plan, "--otlp-traces-endpoint"), "collector:4318"
                )
                plan = self.plan(
                    role,
                    "--model-path",
                    "/models/glm53",
                    "--ib-devices",
                    " rdma2, rdma3 ",
                    environment={"GLM53_IB_DEVICES": "rdma0"},
                )
                self.assertEqual(
                    self.value(plan, "--disaggregation-ib-device"), "rdma2,rdma3"
                )
        self.assert_invalid(
            "prefill", "--model-path", "/models/glm53", "--ib-devices", "rdma0,"
        )

    def test_all_cp_requires_explicit_worker_opt_in_even_with_inherited_env(self):
        for role in ("prefill", "decode"):
            with self.subTest(role=role):
                inherited = {
                    launch.ALL_CP_ENV: "1",
                    "GLM53_MODEL_PATH": "/models/glm53",
                    "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "rust",
                }
                self.assertEqual(
                    self.plan(role, environment=inherited)["env"],
                    {
                        launch.ALL_CP_ENV: "0",
                        "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
                    },
                )
                self.assertEqual(
                    self.plan(role, "--all-cp-ranks-transfer", environment=inherited)[
                        "env"
                    ],
                    {
                        launch.ALL_CP_ENV: "1",
                        "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
                    },
                )

    def test_every_role_preserves_observability_and_has_no_duplicate_flags(self):
        for role in ("router", "prefill", "decode"):
            with self.subTest(role=role):
                options = [] if role == "router" else ["--model-path", "/models/glm53"]
                plan = self.plan(role, *options)
                counts = Counter(arg for arg in plan["argv"] if arg.startswith("--"))
                self.assertTrue(all(count == 1 for count in counts.values()), counts)
                self.assertIn("--enable-trace", plan["argv"])
                self.assertEqual(
                    self.value(plan, "--otlp-traces-endpoint"),
                    launch.DEFAULT_OTLP_ENDPOINT,
                )
                if role != "router":
                    for flag in (
                        "--enable-metrics",
                        "--enable-mfu-metrics",
                        "--enable-metrics-for-all-schedulers",
                    ):
                        self.assertIn(flag, plan["argv"])

    def test_run_execs_the_reviewed_argv_and_inherits_environment(self):
        for role in ("router", "prefill", "decode"):
            with self.subTest(role=role):
                environment = {
                    "GLM53_MODEL_PATH": "/models/path with spaces",
                    "MOONCAKE_MASTER": "store:50051",
                    launch.ALL_CP_ENV: "1",
                    "UNRELATED_SECRET": "do-not-print",
                }
                output = io.StringIO()
                with (
                    patch.dict(os.environ, environment, clear=True),
                    patch.object(launch.os, "execvpe") as execute,
                    contextlib.redirect_stdout(output),
                ):
                    self.assertEqual(launch.main([role, "--run"]), 0)
                plan = json.loads(output.getvalue())
                expected_env = environment | plan["env"]
                execute.assert_called_once_with(
                    plan["argv"][0], plan["argv"], expected_env
                )
                self.assertNotIn("do-not-print", output.getvalue())
                self.assertNotIn("MOONCAKE_MASTER", output.getvalue())
                if role != "router":
                    self.assertEqual(expected_env[launch.ALL_CP_ENV], "0")
                    self.assertEqual(
                        expected_env["SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND"], "python"
                    )

    def test_script_entrypoint_needs_no_sglang_installation(self):
        # -I excludes PYTHONPATH and user site packages; all roles must still dry-run.
        for role in ("router", "prefill", "decode"):
            with self.subTest(role=role):
                result = subprocess.run(
                    [sys.executable, "-I", str(LAUNCH_PATH), role],
                    env={"GLM53_MODEL_PATH": "/not-mounted-on-this-cpu-host"},
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10,
                )
                self.assertEqual(json.loads(result.stdout)["role"], role)
                self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
