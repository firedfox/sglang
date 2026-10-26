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
        values = dict(
            rounds=3,
            run_id="pd1-review",
            output_dir="results/next",
            base_url="http://example.invalid:31180",
            tokenizer="/models/GLM-5.3",
            dataset="/unavailable/dataset.json",
            seed=20261026,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_paired_jobs_isolate_cache_and_counterbalance_profile_order(self):
        plan = plan_module.build_plan(self.options())
        jobs = plan["jobs"]
        self.assertEqual(len(jobs), 12)
        self.assertEqual(len({j["output_file"] for j in jobs}), 12)
        self.assertEqual(len({j["cache_namespace"] for j in jobs}), 6)
        self.assertEqual(
            [j["required_prefill_and_decode_eagle_profile"] for j in jobs[::2]],
            ["5/1/6", "3/1/4", "3/1/4", "5/1/6", "5/1/6", "3/1/4"],
        )
        for fresh, replay in zip(jobs[::2], jobs[1::2]):
            self.assertEqual(
                (fresh["phase"], replay["phase"]), ("fresh_namespace", "replay")
            )
            self.assertEqual(fresh["cache_namespace"], replay["cache_namespace"])
            self.assertTrue(fresh["output_file_must_not_exist"])
        self.assertEqual(plan["expected_prompt_tokens_per_job"], 12582912)
        self.assertEqual(plan["expected_completion_tokens_per_job"], 98304)

    def test_commands_are_literal_and_keep_workload_parameters_fixed(self):
        dataset = "/datasets/space and $(do-not-execute).json"
        jobs = plan_module.build_plan(self.options(dataset=dataset))["jobs"]
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
                plan_module.main(
                    [
                        "--base-url",
                        "http://example.invalid:31180",
                        "--tokenizer",
                        "/models/GLM-5.3",
                        "--dataset",
                        "/missing.json",
                    ]
                ),
                0,
            )
        data = json.loads(output.getvalue())
        self.assertIsNone(data["client_checkout_revision"])
        self.assertEqual(len(data["jobs"]), 12)


if __name__ == "__main__":
    unittest.main()
