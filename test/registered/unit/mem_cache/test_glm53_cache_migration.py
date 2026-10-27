"""CPU regressions for the GLM PD cache stack migrated onto release 0.5.22.

These exercise the release ownership APIs, including the bootstrap FIFO, rather
than replaying old patches against removed cache classes.
"""

import unittest
from array import array
from collections import defaultdict, deque
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.observability.cache_observation import (
    CacheNodeMutationHistory,
    CacheOperationObservation,
    CacheRequestObservation,
    SlowBackupMonitor,
    emit_cache_summary,
)
from sglang.srt.observability.storage_io import StorageIOObserver
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _controller(*, occupied=0, limit=16, enabled=True):
    cc = HybridCacheController.__new__(HybridCacheController)
    cc.host_memory_mode = "cache"
    cc.prefetch_tokens_occupied = occupied
    cc.prefetch_capacity_limit = limit
    cc.prefetch_capacity_retry = enabled
    cc.prefetch_capacity_waiter_rids = []
    cc.prefetch_capacity_waiter_since = {}
    cc.prefetch_capacity_waiter_events = deque(maxlen=64)
    cc.prefetch_capacity_head_events = deque(maxlen=64)
    cc.pp_prefetch_command_group = None
    cc.page_size = 2
    cc.mem_pool_host = SimpleNamespace(
        anchor_entry=SimpleNamespace(
            host_pool=SimpleNamespace(shared_allocation_domain=None)
        )
    )
    return cc


def _cache(cc):
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.tree_core = SimpleNamespace(
        enable_storage=True,
        page_size=2,
        is_eagle=False,
        prefetch_anchor_info=lambda _: (None, None),
    )
    cache.cache_controller = cc
    cache.host_memory_mode = "cache"
    cache.linker = None
    cache.buffer_pipeline = None
    cache.prefetch_threshold = 2
    cache.storage_prefetch_retries = StoragePrefetchRetries()
    cache._storage_prefetch_wait = {}
    cache._prefetch_outcome_stats = defaultdict(int)
    cache.pending_prefetch_requests = {}
    cache.ongoing_prefetch = {}
    cache.prefetch_loaded_tokens_by_reqid = {}
    cache.prefetch_loaded_storage_start_by_reqid = {}
    cache._storage_prefetch_hit_remaining_by_reqid = {}
    cache._prefetch_cost_diagnostics = {}
    return cache


class TestCapacityFIFO(unittest.TestCase):
    def test_exact_capacity_and_strict_fifo(self):
        cc = _controller(occupied=12)
        self.assertTrue(cc.prefetch_rate_limited(8, "head"))
        # This request would fit, but must not bypass the older waiter.
        self.assertTrue(cc.prefetch_rate_limited(2, "tail"))
        cc.prefetch_tokens_occupied = 8
        self.assertFalse(cc.prefetch_rate_limited(8, "head"))
        self.assertTrue(cc.prefetch_rate_limited(2, "tail"))
        cc.prefetch_capacity_waiter_started("head")
        self.assertFalse(cc.prefetch_rate_limited(2, "tail"))
        self.assertEqual(cc.prefetch_capacity_waiter_rids, ["tail"])
        self.assertNotIn("head", cc.prefetch_capacity_waiter_since)

    def test_flag_off_retains_release_gate(self):
        cc = _controller(occupied=12, enabled=False)
        self.assertFalse(cc.prefetch_rate_limited(8, "r"))
        self.assertEqual(cc.prefetch_capacity_waiter_rids, [])
        cc.prefetch_tokens_occupied = 16
        self.assertTrue(cc.prefetch_rate_limited())

    def test_pending_is_request_handle_owned_and_cancel_releases_fifo(self):
        cc = _controller(occupied=12)
        cache = _cache(cc)
        request = CacheRequestHandle("r", 7)
        self.assertIsNone(cache.prefetch_from_storage(request, 0, list(range(8))))
        self.assertEqual(cache.prefetch_capacity_pending_tokens(request), 8)
        self.assertEqual(cc.prefetch_capacity_waiter_rids, ["r"])
        self.assertFalse(cache.storage_prefetch_retries._pending)
        cache.release_aborted_request(request)
        self.assertFalse(cache.pending_prefetch_requests)
        self.assertFalse(cc.prefetch_capacity_waiter_rids)
        # Release is idempotent; no nonexistent operation/host rows are freed.
        cache.release_aborted_request(request)
        self.assertEqual(cc.prefetch_tokens_occupied, 12)

    def test_oversize_falls_back_without_permanent_waiter(self):
        cc = _controller()
        cache = _cache(cc)
        request = CacheRequestHandle("large", 0)
        cache.prefetch_from_storage(request, 0, list(range(20)))
        self.assertFalse(cache.pending_prefetch_requests)
        self.assertFalse(cc.prefetch_capacity_waiter_rids)
        self.assertEqual(cache._prefetch_outcome_stats["declined_host_oversize"], 1)
        self.assertFalse(cache.storage_prefetch_retries._pending)

    def test_bootstrap_head_retries_without_waiting_queue(self):
        cc = _controller(occupied=12)
        cache = _cache(cc)
        head = SimpleNamespace(
            rid="head", cache_request_handle=CacheRequestHandle("head", 0)
        )
        tail = SimpleNamespace(
            rid="tail", cache_request_handle=CacheRequestHandle("tail", 0)
        )
        for req in (head, tail):
            cache.prefetch_from_storage(req.cache_request_handle, 0, list(range(8)))
        cc.prefetch_tokens_occupied = 8
        retry = Mock(
            side_effect=lambda req: cache.release_prefetch_capacity_waiter(
                req.cache_request_handle
            )
        )
        scheduler = SimpleNamespace(
            enable_hicache_storage=True,
            tree_cache=cache,
            waiting_queue=[],
            disagg_prefill_bootstrap_queue=SimpleNamespace(queue=[tail, head]),
            _prefetch_kvcache=retry,
        )
        SchedulerDisaggregationPrefillMixin.retry_hicache_prefetch_capacity_head_in_bootstrap(
            scheduler
        )
        retry.assert_called_once_with(head)
        self.assertEqual(cc.prefetch_capacity_waiter_rids, ["tail"])

    def test_snapshot_is_bounded_and_read_only(self):
        cc = _controller(occupied=16)
        for index in range(100):
            cc.prefetch_rate_limited(2, str(index))
        state = (list(cc.prefetch_capacity_waiter_rids), cc.prefetch_tokens_occupied)
        snapshot = cc.prefetch_capacity_waiter_snapshot()
        self.assertEqual(snapshot["head_rid"], "0")
        self.assertLessEqual(len(snapshot["recent_events"]), 4)
        self.assertLessEqual(len(snapshot["recent_head_events"]), 8)
        self.assertEqual(
            state, (cc.prefetch_capacity_waiter_rids, cc.prefetch_tokens_occupied)
        )


class TestContiguousDevicePrefix(unittest.TestCase):
    def _tree(self):
        params = CacheInitParams(False, None, None, page_size=1)
        component = FullComponent(
            SimpleNamespace(enable_session_radix_cache=False), params
        )
        tree = UnifiedTreeCore(params, {ComponentType.FULL: component})
        tree.enable_hicache = True
        parent = tree.root_node
        nodes = []
        for start, resident in ((0, True), (2, False), (4, True)):
            node = tree._new_node()
            node.parent = parent
            node.key = RadixKey(array("q", range(start, start + 2)))
            node.component_data[ComponentType.FULL].value = (
                torch.arange(start, start + 2) if resident else None
            )
            node.component_data[ComponentType.FULL].host_value = torch.arange(
                start, start + 2
            )
            parent.children[node.key.child_key(tree.page_size)] = node
            nodes.append(node)
            parent = node
        return tree, nodes

    def test_host_only_gap_never_concatenates_device_descendant(self):
        tree, nodes = self._tree()
        result = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", range(6))))
        )
        self.assertEqual(result.device_prefix_len, 2)
        self.assertEqual(result.last_device_node, nodes[0].id)
        self.assertEqual(result.last_host_node, nodes[-1].id)
        self.assertEqual(result.host_hit_length, 4)
        self.assertEqual(result.full_kv_hit_length, 6)

    def test_partial_descendant_split_also_stops_at_gap(self):
        tree, nodes = self._tree()
        result = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", range(5))))
        )
        self.assertEqual(result.device_prefix_len, 2)
        self.assertEqual(result.last_device_node, nodes[0].id)
        self.assertEqual(result.host_hit_length, 3)
        self.assertEqual(result.full_kv_hit_length, 5)

    def test_resident_chain_still_matches_all_tokens(self):
        tree, nodes = self._tree()
        nodes[1].component_data[ComponentType.FULL].value = torch.arange(2, 4)
        result = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", range(6))))
        )
        self.assertEqual(result.device_prefix_len, 6)
        self.assertEqual(result.last_device_node, nodes[-1].id)
        self.assertEqual(result.host_hit_length, 0)


class TestCacheObservations(unittest.TestCase):
    def test_only_committed_admission_changes_match_and_result_emits_once(self):
        history = CacheNodeMutationHistory()
        obs = CacheRequestObservation()
        obs.record_enqueue_match(8, 4, node_ids={"l1": 10, "l2": 11}, history=history)
        with history.eviction_context(
            caller="device_backup",
            allocation_tokens=8,
            requested_evict_tokens=4,
            host_free_before=4,
            component="full",
            trigger_node_id=99,
        ):
            history.record(11, "host_leaf_evicted", 4)
        mutations = obs.admission_node_mutations(4, 0)
        obs.record_admission_match(4, 0, False, "NO_TOKEN", node_mutations=mutations)
        self.assertEqual(obs.admission_match, {})
        obs.record_admission_match(4, 0, True, "CONTINUE", node_mutations=mutations)
        self.assertEqual(obs.admission_match, {"l1": 4, "l2": 0})
        self.assertEqual(obs.node_mutations["events"][0]["caller"], "device_backup")
        obs.collector = Mock()
        obs.log_owner = False
        req = SimpleNamespace(
            cache_observation=obs,
            rid="r",
            _cache_breakdown_computed=True,
            output_ids=[7],
            cached_tokens_device=4,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            finished=lambda: False,
        )
        emit_cache_summary(req, Mock())
        emit_cache_summary(req, Mock())
        self.assertTrue(obs.emitted)
        obs.collector.observe_request.assert_called_once()

    def test_operation_stage_losses_and_terminal_metric_are_once(self):
        obs = CacheOperationObservation(12)
        obs.tokens.update(query_local=10, query=8, allocated=6)
        obs.mark("query_start")
        obs.mark("query_done")
        obs.mark("query_synced")
        obs.mark("allocated")
        obs.mark("get_start")
        obs.mark("get_done")
        operation = SimpleNamespace(cache_observation=obs, host_indices=torch.arange(6))
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.cache_observation_metrics = Mock()
        cache._observe_prefetch(
            operation, local=4, synced=2, duplicate=1, reason="get_partial"
        )
        cache._observe_prefetch(
            operation, local=4, synced=2, duplicate=1, reason="get_partial"
        )
        self.assertEqual(obs.final_tokens["inserted"], 1)
        self.assertEqual(obs.losses["query_rank_min"], 2)
        self.assertEqual(obs.losses["host_allocation_truncated"], 2)
        self.assertEqual(obs.losses["completion_rank_or_pool_min"], 2)
        cache.cache_observation_metrics.observe_operation.assert_called_once()

    def test_mutation_history_detects_reset_and_bounded_eviction(self):
        history = CacheNodeMutationHistory(max_nodes=1, events_per_node=1)
        checkpoint = history.checkpoint()
        history.record(1, "host_leaf_evicted", 2)
        history.record(2, "host_leaf_evicted", 2)
        self.assertEqual(
            history.since(checkpoint, {"l2": 1})["status"], "history_truncated"
        )
        history.reset()
        self.assertEqual(history.since(checkpoint, {"l2": 1})["status"], "tree_reset")

    def test_slow_backup_snapshot_does_not_dequeue_or_repeat_early(self):
        now = [10.0]
        obs = CacheOperationObservation(8, clock=lambda: now[0])
        operation = SimpleNamespace(id=1, cache_observation=obs, completed_tokens=0)
        cc = SimpleNamespace(
            backup_queue=Queue(),
            ack_backup_queue=Queue(),
            active_backup_operation=None,
            write_policy="write_through",
            backup_thread=None,
        )
        cc.backup_queue.put(operation)
        monitor = SlowBackupMonitor(threshold=3, interval=30, clock=lambda: now[0])
        now[0] = 14
        snapshot = monitor.poll(cc)
        self.assertEqual(snapshot["operations"][0]["phase"], "queue")
        self.assertEqual(cc.backup_queue.qsize(), 1)
        now[0] = 20
        self.assertIsNone(monitor.poll(cc))


class TestStorageIOAccounting(unittest.TestCase):
    def test_get_counts_returned_bytes_and_preserves_partial_reply(self):
        observer = StorageIOObserver()
        reply = [3, -1]
        result = observer.call("get", 3, [8, 8, 8], lambda: reply)
        self.assertIs(result, reply)
        values = observer.drain()[0]["get"]
        self.assertEqual(values["requested_objects"], 3)
        self.assertEqual(values["successful_objects"], 1)
        self.assertEqual(values["failed_objects"], 2)
        self.assertEqual(values["successful_bytes"], 3)
        self.assertEqual(observer.drain(), ({}, {}))

    def test_sdk_error_propagates_and_records_all_requested_objects(self):
        observer = StorageIOObserver()
        error = RuntimeError("sdk")
        with self.assertRaisesRegex(RuntimeError, "sdk"):
            observer.call("put", 2, [[2, 4], [3, 5]], Mock(side_effect=error))
        values = observer.drain()[0]["put"]
        self.assertEqual(values["failed_objects"], 2)
        self.assertEqual(values["successful_bytes"], 0)
        self.assertEqual(values["exceptions"], 1)

    def test_mooncake_multi_buffer_group_config_and_reply_unchanged(self):
        store = MooncakeStore.__new__(MooncakeStore)
        store._io_observer = StorageIOObserver()
        store._can_use_group_semantics = lambda: True
        store._replicate_config_cls = SimpleNamespace
        reply = [0, -1]
        store.store = SimpleNamespace(
            batch_put_from_multi_buffers=Mock(return_value=reply)
        )
        keys, ptrs, sizes = ["a", "b"], [[10, 20], [30, 40]], [[2, 4], [3, 5]]
        result = store._put_batch_zero_copy_impl(
            keys, ptrs, sizes, ["ga", "gb"], pool="kv"
        )
        self.assertIs(result, reply)
        args = store.store.batch_put_from_multi_buffers.call_args.args
        self.assertEqual(args[:3], (keys, ptrs, sizes))
        self.assertEqual(args[3].group_ids, ["ga", "gb"])
        values = store._io_observer.drain()[0]["put"]
        self.assertEqual(values["successful_bytes"], 6)
        self.assertEqual(values["partial_calls"], 1)


if __name__ == "__main__":
    unittest.main()
