# SPDX-License-Identifier: Apache-2.0
"""Cost-aware prefetch calibration and runtime policy validation on CPU."""

import argparse
import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_MODEL = '{"recompute_tokens_per_second":1000,"read_tokens_per_second":10000}'


class TestHiCachePrefetchCostArgs(unittest.TestCase):
    def args(self, **overrides):
        fields = dict(
            model_path="dummy",
            enable_hierarchical_cache=True,
            hicache_storage_prefetch_policy="cost_aware",
            hicache_storage_prefetch_cost_model=_MODEL,
        )
        fields.update(overrides)
        return ServerArgs(**fields)

    def test_default_policy_and_calibration_do_not_enable_cost_decisions(self):
        args = ServerArgs(model_path="dummy")
        args.resolve_once()
        self.assertEqual(
            resolution_result(args, "hicache_storage_prefetch_policy"), "timeout"
        )
        self.assertIsNone(
            resolution_result(args, "hicache_storage_prefetch_cost_model")
        )

    def test_cli_registers_policy_and_model(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = ServerArgs.from_cli_args(
            parser.parse_args(
                [
                    "--model-path",
                    "dummy",
                    "--enable-hierarchical-cache",
                    "--hicache-storage-prefetch-policy",
                    "cost_aware",
                    "--hicache-storage-prefetch-cost-model",
                    _MODEL,
                ]
            )
        )
        args.resolve_once()
        self.assertEqual(args.hicache_storage_prefetch_policy, "cost_aware")
        self.assertEqual(args.hicache_storage_prefetch_cost_model, _MODEL)

    def test_invalid_calibration_fails_before_dummy_model_short_circuit(self):
        for model in ("not-json", "{}", '{"recompute_tokens_per_second":false}'):
            with self.subTest(model=model):
                with self.assertRaisesRegex(ValueError, "prefetch-cost-model"):
                    self.args(hicache_storage_prefetch_cost_model=model).resolve_once()

    def test_missing_model_keeps_explicit_timeout_fallback(self):
        with self.assertLogs(
            "sglang.srt.arg_groups.hicache_hook", level="WARNING"
        ) as logs:
            args = self.args(hicache_storage_prefetch_cost_model=None)
            args.resolve_once()
        self.assertIn("uses the timeout policy", " ".join(logs.output))

    def test_unsupported_backends_cannot_silently_ignore_the_policy(self):
        for config in (
            {"enable_hierarchical_cache": False},
            {"disable_radix_cache": True},
            {"radix_cache_backend": "custom"},
            {"enable_unified_cache_external_linker": True},
            {"pp_size": 2, "hicache_host_memory_mode": "buffer_only"},
        ):
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, "cost_aware"):
                    self.args(**config).resolve_once()

    def test_tp_cp_and_cache_mode_pp_keep_the_supported_stop_path(self):
        for config in (
            {"tp_size": 8, "attn_cp_size": 8},
            {"pp_size": 2, "hicache_host_memory_mode": "cache"},
        ):
            with self.subTest(config=config):
                self.args(**config).resolve_once()

    def test_runtime_attach_accepts_cost_policy_and_retains_launch_calibration(self):
        calibration = object()
        cache = SimpleNamespace(
            prefetch_stop_policy="timeout",
            prefetch_cost_model=calibration,
            host_memory_mode="cache",
            pp_size=1,
            is_write_back=False,
            enable_storage=True,
            cache_controller=SimpleNamespace(storage_backend_type="file"),
        )
        ok, _ = StorageAttachment(cache).attach(
            "file", hicache_storage_prefetch_policy="cost_aware"
        )
        self.assertTrue(ok)
        self.assertEqual(cache.prefetch_stop_policy, "cost_aware")
        self.assertIs(cache.prefetch_cost_model, calibration)

    def test_runtime_pp_ticket_rejection_has_no_side_effects(self):
        cache = SimpleNamespace(
            prefetch_stop_policy="timeout", host_memory_mode="buffer_only", pp_size=2
        )
        before = vars(cache).copy()
        ok, message = StorageAttachment(cache).attach(
            "file", hicache_storage_prefetch_policy="cost_aware"
        )
        self.assertFalse(ok)
        self.assertIn("PP buffer_only", message)
        self.assertEqual(vars(cache), before)


if __name__ == "__main__":
    unittest.main()
