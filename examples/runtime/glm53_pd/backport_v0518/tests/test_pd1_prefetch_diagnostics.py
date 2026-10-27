"""Exercise the deployed legacy consensus and terminal observation hooks on CPU."""

import json
import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from prometheus_client import CollectorRegistry

import sglang.srt.mem_cache.unified_radix_cache as cache_module
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import PrefetchOperation
from sglang.srt.mem_cache.prefetch_cost import PrefetchCostModel
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.observability.cache_metrics import CacheMetricsCollector


class TestPD1PrefetchDiagnostics(unittest.TestCase):
    def fixture(self, *, completed=34000, elapsed=1.0, independent=False):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(page_size=64)
        cache.prefetch_stop_policy = "cost_aware"
        cache.prefetch_cost_model = PrefetchCostModel(
            recompute_tokens_per_second=47000,
            min_observation_seconds=0.2,
            cancel_margin=1.2,
        )
        cache.prefetch_timeout_base = 1.0
        cache.prefetch_timeout_per_page = 0.25
        cache._all_reduce_attn_groups = lambda states, op: None
        cache.cache_observation_metrics = None
        transfer = SimpleNamespace(
            name=PoolName.MAMBA if independent else PoolName.INDEXER,
            indices_from_pool=None if independent else PoolName.KV,
            keys=["page"] * 2048,
        )
        operation = PrefetchOperation(
            "pd1-sample", [1] * 131072, pool_transfers=[transfer]
        )
        operation.start_time = 100.0
        operation.hash_value = ["page"] * 2048
        operation.host_indices = torch.empty(131072, dtype=torch.int64)
        operation.completed_tokens = completed
        operation._prefetch_progress = (131072, 0, 100.0)
        operation.cache_observation.tokens.update(query=131072, allocated=131072)
        operation.cache_observation.clock = lambda: 100.0 + elapsed
        return cache, operation, 100.0 + elapsed

    def poll(self, cache, operation, now):
        with patch.object(cache_module.time, "monotonic", return_value=now):
            return cache.can_terminate_prefetch(operation)

    def test_deployed_model_explains_valid_34k_cancel_and_one_terminal_event(self):
        cache, operation, now = self.fixture()
        registry = CollectorRegistry()
        cache.cache_observation_metrics = CacheMetricsCollector({"rank": "0"}, registry)
        with self.assertLogs(cache_module.logger, logging.DEBUG) as logs:
            self.assertTrue(self.poll(cache, operation, now))
            cache._observe_prefetch(operation, 34000, 34000, 0)
            cache._observe_prefetch(operation, 34000, 34000, 0)
            self.poll(cache, operation, now)
        records = [
            r
            for r in logs.records
            if r.getMessage().startswith("HiCache prefetch cost decision ")
        ]
        self.assertEqual(len(records), 1)
        data = json.loads(records[0].getMessage().split("decision ", 1)[1])
        estimate = data["last_evaluation"]
        self.assertEqual(estimate["read_rate_source"], "observed")
        self.assertEqual(estimate["read_tokens_per_second"], 34000)
        self.assertIsNone(estimate["configured_read_tokens_per_second"])
        self.assertEqual(estimate["remaining_tokens"], estimate["recompute_tokens"])
        self.assertEqual(estimate["decision"], "cancel")
        self.assertEqual(data["local_vote"], "cost_aware")
        self.assertEqual(data["agreed_stop_reason"], "cost_aware")
        self.assertEqual(data["terminal"], "cost_aware")
        self.assertEqual(data["reuse_mode"], "partial")
        self.assertFalse(hasattr(operation, "_prefetch_cost_diagnostic"))
        self.assertEqual(
            registry.get_sample_value(
                "sglang:cache_operations_total",
                {"rank": "0", "operation": "prefetch", "reason": "cost_aware"},
            ),
            1,
        )

    def test_short_and_zero_progress_samples_are_not_fake_keep_estimates(self):
        for completed, elapsed, reason in (
            (3400, 0.1, "observation_too_short"),
            (0, 1.0, "no_read_rate"),
        ):
            with self.subTest(reason=reason):
                cache, operation, now = self.fixture(
                    completed=completed, elapsed=elapsed
                )
                with patch.object(
                    cache_module.logger, "isEnabledFor", return_value=True
                ):
                    self.assertFalse(self.poll(cache, operation, now))
                diagnostic = operation._prefetch_cost_diagnostic
                self.assertEqual(
                    diagnostic["last_evaluation"]["decision"], "timeout_fallback"
                )
                self.assertEqual(diagnostic["fallback_reason"], reason)

    def test_effective_fast_read_and_independent_sidecar_can_keep_prefetch(self):
        for options, reuse in (
            (dict(completed=50000), "partial"),
            (dict(independent=True), "all_or_nothing"),
        ):
            with self.subTest(options=options):
                cache, operation, now = self.fixture(**options)
                with self.assertLogs(cache_module.logger, logging.DEBUG) as logs:
                    self.assertFalse(self.poll(cache, operation, now))
                    cache._observe_prefetch(operation, reason="cancelled")
                data = json.loads(logs.records[0].getMessage().split("decision ", 1)[1])
                self.assertEqual(data["last_evaluation"]["decision"], "keep")
                self.assertEqual(data["reuse_mode"], reuse)

    def test_local_cost_vote_does_not_override_legacy_peer_readiness(self):
        cache, operation, now = self.fixture()
        cache._all_reduce_attn_groups = lambda states, op: states.__setitem__(0, 1)
        with patch.object(cache_module.logger, "isEnabledFor", return_value=True):
            self.assertFalse(self.poll(cache, operation, now))
        diagnostic = operation._prefetch_cost_diagnostic
        self.assertEqual(diagnostic["local_vote"], "cost_aware")
        self.assertFalse(diagnostic["agreed_can_terminate"])
        self.assertNotIn("agreed_stop_reason", diagnostic)
        self.assertEqual(
            diagnostic["consensus_rule"], "all_ranks_ready_or_any_terminated"
        )
        self.assertFalse(hasattr(operation.cache_observation, "stop_reason"))

    def test_peer_termination_is_distinct_from_local_keep_vote(self):
        cache, operation, now = self.fixture(completed=50000)
        cache._all_reduce_attn_groups = lambda states, op: states.__setitem__(1, 1)
        with patch.object(cache_module.logger, "isEnabledFor", return_value=True):
            self.assertTrue(self.poll(cache, operation, now))
        diagnostic = operation._prefetch_cost_diagnostic
        self.assertEqual(diagnostic["local_vote"], "keep_waiting")
        self.assertEqual(diagnostic["agreed_stop_reason"], "peer_terminated")

    def test_deadline_is_reported_without_a_fabricated_cost_sample(self):
        cache, operation, now = self.fixture()
        operation.start_time = -1000.0
        with self.assertLogs(cache_module.logger, logging.DEBUG) as logs:
            self.assertTrue(self.poll(cache, operation, now))
            cache._observe_prefetch(operation, 34000, 34000, 0)
        data = json.loads(logs.records[0].getMessage().split("decision ", 1)[1])
        self.assertEqual(data["local_vote"], "deadline")
        self.assertEqual(data["agreed_stop_reason"], "timeout")
        self.assertEqual(data["terminal"], "timeout")
        self.assertIsNone(data["last_evaluation"])

    def test_other_stop_policy_has_no_cost_diagnostic_state(self):
        cache, operation, now = self.fixture()
        cache.prefetch_stop_policy = "wait_complete"
        with (
            patch.object(cache_module.logger, "isEnabledFor", return_value=True),
            patch.object(cache_module.logger, "debug") as debug,
        ):
            self.assertFalse(self.poll(cache, operation, now))
            cache._observe_prefetch(operation, reason="cancelled")
        self.assertFalse(hasattr(operation, "_prefetch_cost_diagnostic"))
        debug.assert_not_called()

    def test_complete_kv_waiting_for_sidecar_does_not_invent_cost_evaluation(self):
        cache, operation, now = self.fixture(completed=131072)
        with self.assertLogs(cache_module.logger, logging.DEBUG) as logs:
            self.assertFalse(self.poll(cache, operation, now))
            operation.pool_transfers_done = True
            self.assertTrue(self.poll(cache, operation, now))
            cache._observe_prefetch(operation, 131072, 131072, 0)
        data = json.loads(logs.records[0].getMessage().split("decision ", 1)[1])
        self.assertIsNone(data["last_evaluation"])
        self.assertEqual(data["terminal"], "complete")
        self.assertEqual(data["agreed_stop_reason"], "complete")

    def test_info_path_does_not_construct_diagnostic_estimates(self):
        cache, operation, now = self.fixture()
        with (
            patch.object(cache_module.logger, "isEnabledFor", return_value=False),
            patch.object(
                PrefetchCostModel,
                "describe_estimate",
                side_effect=AssertionError("DEBUG only"),
            ),
        ):
            self.assertTrue(self.poll(cache, operation, now))
            cache._observe_prefetch(operation, 34000, 34000, 0)
        self.assertFalse(hasattr(operation, "_prefetch_cost_diagnostic"))

    def test_diagnostic_cleanup_when_log_level_changes_before_terminal(self):
        cache, operation, now = self.fixture()
        with patch.object(cache_module.logger, "isEnabledFor", return_value=True):
            self.assertTrue(self.poll(cache, operation, now))
        self.assertTrue(hasattr(operation, "_prefetch_cost_diagnostic"))
        with patch.object(cache_module.logger, "isEnabledFor", return_value=False):
            cache._observe_prefetch(operation, 34000, 34000, 0)
        self.assertFalse(hasattr(operation, "_prefetch_cost_diagnostic"))


if __name__ == "__main__":
    unittest.main()
