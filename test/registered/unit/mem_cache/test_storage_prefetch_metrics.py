# SPDX-License-Identifier: Apache-2.0
"""Cumulative prefetch outcomes reach Prometheus without poll overcounting."""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.observability.metrics_collector import (
    StorageMetrics,
    StorageMetricsCollector,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestStoragePrefetchMetrics(unittest.TestCase):
    def setUp(self):
        self.registry = CollectorRegistry()

        class Collector(StorageMetricsCollector):
            _counter_cls = staticmethod(partial(Counter, registry=self.registry))
            _gauge_cls = staticmethod(partial(Gauge, registry=self.registry))
            _histogram_cls = staticmethod(partial(Histogram, registry=self.registry))

        self.collector = Collector({"model_name": "test", "tp_rank": "0"})

    def outcome(self, name):
        return self.registry.get_sample_value(
            "sglang:storage_prefetch_outcomes_total",
            {"model_name": "test", "tp_rank": "0", "outcome": name},
        )

    def test_repeated_snapshots_export_each_agreed_stop_once(self):
        metrics = StorageMetrics(
            prefetch_stats={"stopped_cost_aware": 1, "stopped_deadline": 0}
        )
        for _ in range(20):
            self.collector.log_storage_metrics(metrics)
        self.assertEqual(self.outcome("stopped_cost_aware"), 1)
        self.assertEqual(self.outcome("stopped_deadline"), 0)
        metrics.prefetch_stats = {"stopped_cost_aware": 2, "stopped_deadline": 1}
        self.collector.log_storage_metrics(metrics)
        self.assertEqual(self.outcome("stopped_cost_aware"), 2)
        self.assertEqual(self.outcome("stopped_deadline"), 1)

    def test_outcome_labels_are_bounded_and_exclude_token_counts(self):
        self.collector.log_storage_metrics(
            StorageMetrics(
                prefetch_stats={
                    "issued": 4,
                    "declined_host_oversize": 2,
                    "l3_miss_tokens": 123456,
                    "request-user-supplied-id": 99,
                }
            )
        )
        self.assertEqual(self.outcome("issued"), 4)
        self.assertEqual(self.outcome("declined_host_oversize"), 2)
        observed = {
            sample.labels["outcome"]
            for metric in self.registry.collect()
            for sample in metric.samples
            if sample.name == "sglang:storage_prefetch_outcomes_total"
        }
        self.assertEqual(observed, set(StorageMetricsCollector.PREFETCH_OUTCOMES))
        self.assertIsNone(self.outcome("l3_miss_tokens"))

    def test_controller_reset_starts_new_series_without_decrement(self):
        for value in (3, 3, 0, 1, 1, 2):
            self.collector.log_prefetch_outcomes({"stopped_cost_aware": value})
        self.assertEqual(self.outcome("stopped_cost_aware"), 5)

    def test_missing_or_invalid_values_do_not_reset_previous_snapshot(self):
        self.collector.log_prefetch_outcomes({"issued": 3})
        self.collector.log_storage_metrics(StorageMetrics())
        for value in (None, True, -1, "3", float("nan"), float("inf"), 10**400):
            with self.subTest(value=value):
                self.collector.log_prefetch_outcomes({"issued": value})
        self.collector.log_prefetch_outcomes({"issued": 3})
        self.assertEqual(self.outcome("issued"), 3)

    def test_existing_io_histograms_still_receive_samples(self):
        self.collector.log_storage_metrics(
            StorageMetrics(prefetch_pgs=[4, 8], prefetch_bandwidth=[1.5])
        )
        labels = {"model_name": "test", "tp_rank": "0"}
        self.assertEqual(
            self.registry.get_sample_value("sglang:prefetch_pgs_count", labels), 2
        )
        self.assertEqual(
            self.registry.get_sample_value("sglang:prefetch_pgs_sum", labels), 12
        )
        self.assertEqual(
            self.registry.get_sample_value("sglang:prefetch_bandwidth_sum", labels), 1.5
        )

    def test_controller_refreshes_snapshot_when_backend_reuses_metrics_object(self):
        metrics = StorageMetrics()
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(enable_storage=False)
        cache.linker = None
        cache.buffer_pipeline = None
        cache.enable_storage_metrics = True
        cache.storage_metrics_collector = self.collector
        cache.cache_controller = SimpleNamespace(
            storage_backend=SimpleNamespace(get_stats=lambda: metrics)
        )
        cache._drain_async_work = Mock()
        cache.flush_pending_backups = Mock()
        cache._sync_hicache_ready_counts = Mock(return_value=(0, 0, [], []))
        cache.writing_check = Mock()
        cache.loading_check = Mock()
        cache._prefetch_outcome_stats = {"stopped_cost_aware": 1}
        cache.check_hicache_events()
        cache._prefetch_outcome_stats["stopped_cost_aware"] = 2
        cache.check_hicache_events()
        cache.check_hicache_events()
        self.assertEqual(self.outcome("stopped_cost_aware"), 2)


if __name__ == "__main__":
    unittest.main()
