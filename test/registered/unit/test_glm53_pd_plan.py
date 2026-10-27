"""Offline checks for cache isolation and reproducible GLM-5.3 comparison plans."""

import contextlib
import hashlib
import importlib.util
import io
import json
import runpy
import shlex
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "examples/runtime/glm53_pd/plan_next_round.py"
runpy.run_path(str(REPO_ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
](est_time=1, suite="base-a-test-cpu")
spec = importlib.util.spec_from_file_location("glm53_pd_plan", SCRIPT)
plan_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan_module)


class TestGLM53PDPlan(unittest.TestCase):
    def options(self, **overrides):
        values = {
            "rounds": 3,
            "run_id": "pd1-review",
            "output_dir": "results/next",
            "base_url": "http://example.invalid:31180",
            "tokenizer": "/models/GLM-5.3",
            "workload": "native-random",
            "dataset": "/unavailable/dataset.json",
            "seed": 20261026,
            "gsp_num_groups": None,
            "gsp_prompts_per_group": None,
            "gsp_system_prompt_len": None,
            "gsp_question_len": None,
            "gsp_range_ratio": None,
            "gsp_fast_prepare": False,
            "gsp_ordered": False,
            "gsp_send_routing_key": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def chat_options(self, **overrides):
        # Deliberate test fixture values, not reconstructed deployment inputs.
        values = {
            "workload": "chat-gsp",
            "dataset": None,
            "gsp_num_groups": 6,
            "gsp_prompts_per_group": 16,
            "gsp_system_prompt_len": 4096,
            "gsp_question_len": 128,
            "gsp_range_ratio": 1.0,
        }
        values.update(overrides)
        return self.options(**values)

    def cli_args(self, workload):
        args = [
            "--workload",
            workload,
            "--base-url",
            "http://example.invalid:31180",
            "--tokenizer",
            "/models/GLM-5.3",
            "--seed",
            "12345",
        ]
        if workload == "native-random":
            args += ["--dataset", "/missing.json"]
        else:
            args += [
                "--gsp-num-groups",
                "6",
                "--gsp-prompts-per-group",
                "16",
                "--gsp-system-prompt-len",
                "4096",
                "--gsp-question-len",
                "128",
                "--gsp-range-ratio",
                "1",
            ]
        return args

    def assert_invalid(self, args):
        error = io.StringIO()
        with (
            contextlib.redirect_stderr(error),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            plan_module.main(args)
        self.assertEqual(raised.exception.code, 2)
        return error.getvalue()

    def test_paired_jobs_isolate_cache_and_counterbalance_profile_order(self):
        for options in (self.options(), self.chat_options()):
            with self.subTest(workload=options.workload):
                plan = plan_module.build_plan(options)
                jobs = plan["jobs"]
                self.assertEqual(plan["schema_version"], 2)
                self.assertFalse(plan["original_inputs_reconstructed"])
                self.assertEqual(len(jobs), 12)
                self.assertEqual(len({j["output_file"] for j in jobs}), 12)
                self.assertEqual(len({j["cache_namespace"] for j in jobs}), 6)
                self.assertEqual(
                    [j["required_decode_eagle_profile"] for j in jobs[::2]],
                    ["5/1/6", "3/1/4", "3/1/4", "5/1/6", "5/1/6", "3/1/4"],
                )
                for fresh, replay in zip(jobs[::2], jobs[1::2]):
                    self.assertEqual(
                        (fresh["phase"], replay["phase"]), ("fresh_namespace", "replay")
                    )
                    self.assertEqual(
                        fresh["cache_namespace"], replay["cache_namespace"]
                    )
                    self.assertTrue(fresh["output_file_must_not_exist"])
                for job in jobs:
                    self.assertEqual(
                        job["prefill_configuration"], "preserve_validated_deployment"
                    )
                    self.assertNotIn("required_prefill_and_decode_eagle_profile", job)
                self.assertEqual(plan["expected_completion_tokens_per_job"], 98304)

    def test_chat_workload_preserves_explicit_shape_but_input_usage_is_unknown(self):
        options = self.chat_options(
            gsp_fast_prepare=True, gsp_ordered=True, gsp_send_routing_key=True
        )
        plan = plan_module.build_plan(options)
        self.assertIsNone(plan["expected_prompt_tokens_per_job"])
        self.assertEqual(
            plan["expected_prompt_tokens_source"],
            "unknown_until_server_usage_is_recorded",
        )
        self.assertEqual(
            plan["comparison_scope"], "same_reported_backend_and_dataset_family"
        )
        expected = {
            "--backend": "sglang-oai-chat",
            "--dataset-name": "generated-shared-prefix",
            "--gsp-num-groups": "6",
            "--gsp-prompts-per-group": "16",
            "--gsp-system-prompt-len": "4096",
            "--gsp-question-len": "128",
            "--gsp-range-ratio": "1.0",
            "--gsp-output-len": "1024",
            "--gsp-num-turns": "1",
            "--gsp-group-distribution": "uniform",
            "--max-concurrency": "32",
            "--num-prompts": "96",
        }
        for job in plan["jobs"]:
            argv = job["argv"]
            self.assertEqual(shlex.split(job["shell_command"]), argv)
            for flag, value in expected.items():
                self.assertEqual(argv[argv.index(flag) + 1], value)
            for flag in (
                "--gsp-fast-prepare",
                "--gsp-ordered",
                "--gsp-send-routing-key",
            ):
                self.assertIn(flag, argv)
                self.assertTrue(
                    plan["generation_parameters"][flag[2:].replace("-", "_")]
                )
            self.assertNotIn("--tokenize-prompt", argv)
            self.assertNotIn("--dataset-path", argv)
            self.assertNotIn("--random-input-len", argv)
        self.assertIsNone(plan["dataset_sha256"])

    def test_missing_workload_seed_or_gsp_shape_is_rejected(self):
        for workload in ("native-random", "chat-gsp"):
            for flag in ("--workload", "--seed"):
                with self.subTest(workload=workload, missing=flag):
                    args = self.cli_args(workload)
                    index = args.index(flag)
                    del args[index : index + 2]
                    self.assert_invalid(args)
        for flag in (
            "--gsp-num-groups",
            "--gsp-prompts-per-group",
            "--gsp-system-prompt-len",
            "--gsp-question-len",
            "--gsp-range-ratio",
        ):
            with self.subTest(missing=flag):
                args = self.cli_args("chat-gsp")
                index = args.index(flag)
                del args[index : index + 2]
                self.assertIn(
                    "requires explicit generation settings", self.assert_invalid(args)
                )

    def test_fixed_output_and_group_size_constraints_are_enforced(self):
        for flag, value in (
            ("--gsp-num-groups", "7"),
            ("--gsp-num-groups", "0"),
            ("--gsp-range-ratio", "0.5"),
            ("--gsp-range-ratio", "nan"),
            ("--gsp-range-ratio", "inf"),
            ("--gsp-range-ratio", "-1"),
            ("--gsp-system-prompt-len", "-1"),
            ("--seed", "-1"),
            ("--seed", str(2**32)),
        ):
            with self.subTest(flag=flag, value=value):
                self.assert_invalid(self.cli_args("chat-gsp") + [flag, value])
        self.assertIn(
            "randomize",
            self.assert_invalid(
                self.cli_args("chat-gsp") + ["--gsp-range-ratio", "0.5"]
            ),
        )

    def test_workload_specific_arguments_cannot_be_silently_ignored(self):
        self.assert_invalid(self.cli_args("chat-gsp") + ["--dataset", "/unused.json"])
        self.assert_invalid(self.cli_args("native-random") + ["--gsp-num-groups", "6"])
        self.assert_invalid(self.cli_args("native-random") + ["--gsp-fast-prepare"])
        native = self.cli_args("native-random")
        self.assertIn("requires --dataset", self.assert_invalid(native[:-2]))

    def test_same_run_id_keeps_different_workloads_in_separate_namespaces(self):
        native = plan_module.build_plan(self.options())
        chat = plan_module.build_plan(self.chat_options())
        for field in ("cache_namespace", "output_file"):
            self.assertFalse(
                {j[field] for j in native["jobs"]} & {j[field] for j in chat["jobs"]}
            )

    def test_commands_are_literal_and_keep_workload_parameters_fixed(self):
        dataset = "/datasets/space and $(do-not-execute).json"
        plan = plan_module.build_plan(self.options(dataset=dataset))
        jobs = plan["jobs"]
        self.assertEqual(plan["comparison_scope"], "independent_baseline")
        self.assertEqual(plan["expected_prompt_tokens_per_job"], 12582912)
        self.assertEqual(
            plan["expected_prompt_tokens_source"], "fixed_native_input_ids"
        )
        expected = {
            "--num-prompts": "96",
            "--random-input-len": "131072",
            "--random-output-len": "1024",
            "--max-concurrency": "32",
            "--random-range-ratio": "1",
            "--warmup-requests": "0",
            "--seed": "20261026",
            "--temperature": "0",
            "--dataset-path": dataset,
        }
        for job in jobs:
            argv = job["argv"]
            self.assertEqual(shlex.split(job["shell_command"]), argv)
            for flag, value in expected.items():
                self.assertEqual(argv[argv.index(flag) + 1], value)
            self.assertNotIn("--profile", argv)
            self.assertNotIn("--gsp-num-groups", argv)
            self.assertEqual(
                json.loads(argv[argv.index("--extra-request-body") + 1]),
                {"extra_key": job["cache_namespace"]},
            )

    def test_dataset_digest_covers_actual_bytes_and_missing_file_stays_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "sample.json"
            payload = b'[{"conversations": []}]\n'
            dataset.write_bytes(payload)
            plan = plan_module.build_plan(self.options(dataset=str(dataset)))
            self.assertEqual(
                plan["dataset_sha256"], hashlib.sha256(payload).hexdigest()
            )
        self.assertIsNone(plan_module.build_plan(self.options())["dataset_sha256"])

    def test_cli_remains_offline_and_without_git_provenance_is_unknown(self):
        output = io.StringIO()
        with (
            patch.object(plan_module.subprocess, "run", side_effect=FileNotFoundError),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(
                plan_module.main(self.cli_args("chat-gsp")),
                0,
            )
        data = json.loads(output.getvalue())
        self.assertIsNone(data["client_checkout_revision"])
        self.assertEqual(data["seed"], 12345)
        self.assertIsNone(data["expected_prompt_tokens_per_job"])
        self.assertEqual(len(data["jobs"]), 12)


if __name__ == "__main__":
    unittest.main()
