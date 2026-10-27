"""CPU control-flow regressions for the v0.5.18 pd1 diagnostic backport."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler_components.metrics_reporter import PrefillStats
from sglang.srt.managers.scheduler_components.prefill_time_budget import (
    PrefillTimeBudget,
    PrefillTimeModel,
)
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils.common import Range


class TestPD1PrefillDiagnostics(unittest.TestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def fixture(
        self, *, budget_ms=128, chunk=256, input_limit=4096, prefix=0, remaining=512
    ):
        tree = MagicMock()
        tree.disable = False
        tree.is_tree_cache.return_value = False
        tree.full_evictable_size.return_value = 0
        tree.swa_evictable_size.return_value = 0
        tree.evictable_size.return_value = 0
        tree.inc_lock_ref.return_value = IncLockRefResult()
        tree.dec_lock_ref.return_value = DecLockRefResult()
        allocator = MagicMock()
        allocator.full_available_size.return_value = 1_000_000
        allocator.swa_available_size.return_value = 1_000_000
        allocator.available_size.return_value = 1_000_000
        running = MagicMock()
        running.reqs = []
        budget = (
            PrefillTimeBudget(PrefillTimeModel(linear_ms_per_token=1), budget_ms)
            if budget_ms is not None
            else None
        )
        adder = PrefillAdder(
            page_size=64,
            tree_cache=tree,
            token_to_kv_pool_allocator=allocator,
            running_batch=running,
            new_token_ratio=1.0,
            rem_input_tokens=input_limit,
            rem_chunk_tokens=chunk,
            num_mixed_decode_tokens=0,
            priority_scheduling_preemption_threshold=0,
            prefill_time_budget=budget,
        )
        req = MagicMock(spec=Req)
        req.rid = "pd1-diagnostic"
        req.prefix_indices = list(range(prefix))
        req.full_untruncated_fill_ids = list(range(prefix + remaining))
        req.origin_input_ids = list(req.full_untruncated_fill_ids)
        req.output_ids = []
        req.retracted_stain = False
        req.host_hit_length = 0
        req.storage_hit_length = 0
        req.last_node = MagicMock()
        req.sampling_params = SimpleNamespace(max_new_tokens=16, ignore_eos=False)
        req.needs_host_load_back.return_value = False
        req.extend_range = Range(prefix, prefix + remaining)
        req.set_extend_range = MagicMock(
            side_effect=lambda start, end: setattr(
                req, "extend_range", Range(start, end)
            )
        )
        return adder, req

    def admit(self, adder, req, path):
        if path == "ordinary":
            return adder.add_one_req(
                req, has_chunked_req=False, truncation_align_size=None
            )
        if path == "continuation":
            return adder.add_chunked_req(req)
        req.sampling_params.ignore_eos = True
        return adder.add_one_req_ignore_eos(req)

    def test_all_admission_paths_count_real_time_cuts(self):
        for path in ("ordinary", "continuation", "ignore_eos"):
            with self.subTest(path=path):
                adder, req = self.fixture()
                self.admit(adder, req, path)
                self.assertEqual(req.extend_range.length, 128)
                self.assertEqual(adder.prefill_time_limited_requests, 1)
                self.assertEqual(adder.prefill_time_overrun_requests, 0)
                self.assertEqual(adder.prefill_time_budget.admitted_requests, 1)

    def test_input_cap_and_alignment_are_not_counted_as_time_cuts(self):
        for path in ("ordinary", "continuation", "ignore_eos"):
            for options in (
                dict(budget_ms=512, input_limit=128),
                dict(budget_ms=130, chunk=190),
            ):
                with self.subTest(path=path, options=options):
                    adder, req = self.fixture(**options)
                    self.admit(adder, req, path)
                    self.assertEqual(req.extend_range.length, 128)
                    self.assertEqual(adder.prefill_time_limited_requests, 0)

    def test_candidate_rejected_after_selection_does_not_spend_or_count(self):
        adder, req = self.fixture()
        with patch.object(
            adder, "_check_prefill_tile_budget", return_value=AddReqResult.OTHER
        ):
            result = self.admit(adder, req, "ordinary")
        self.assertIs(result, AddReqResult.OTHER)
        self.assertEqual(adder.can_run_list, [])
        self.assertEqual(adder.prefill_time_budget.admitted_requests, 0)
        self.assertEqual(adder.prefill_time_limited_requests, 0)
        self.assertEqual(adder.prefill_time_overrun_requests, 0)
        self.admit(adder, req, "ordinary")
        self.assertEqual(adder.prefill_time_limited_requests, 1)

    def test_minimum_progress_overrun_is_reported_only_after_commit(self):
        adder, req = self.fixture(budget_ms=1)
        self.admit(adder, req, "ordinary")
        self.assertEqual(req.extend_range.length, 64)
        self.assertEqual(adder.prefill_time_overrun_requests, 1)
        self.assertEqual(adder.prefill_time_budget.used_ms, 64)

    def test_prefill_stats_snapshot_uses_old_extend_range_before_mutation(self):
        adder, req = self.fixture(prefix=64)
        self.admit(adder, req, "ordinary")
        stats = PrefillStats.from_adder(adder, [])
        req.set_extend_range(192, 512)
        self.assertEqual(stats.time_budget_ms, 128)
        self.assertEqual(stats.time_estimated_ms, 128)
        self.assertEqual(stats.time_limited_requests, 1)
        self.assertEqual(stats.time_sum_extend, 128)
        self.assertEqual(stats.time_sum_prefix_extend, 64 * 128)
        self.assertEqual(stats.time_sum_extend_squared, 128 * 128)

    def test_default_off_does_not_add_estimates_or_change_shape(self):
        adder, req = self.fixture(budget_ms=None)
        self.admit(adder, req, "ordinary")
        stats = PrefillStats.from_adder(adder, [])
        self.assertEqual(req.extend_range.length, 256)
        self.assertIsNone(stats.time_budget_ms)
        self.assertIsNone(stats.time_estimated_ms)
        self.assertEqual(stats.time_limited_requests, 0)

    def test_deployed_model_predicts_a_cold_128k_time_cut(self):
        model = PrefillTimeModel(120, 0.001, 1.2e-7)
        budget = PrefillTimeBudget(model, 1500)
        self.assertAlmostEqual(
            model.fixed_ms + model.token_cost_ms(0, 131072), 2312.65630208
        )
        selected = budget.select_tokens(
            prefix_len=0,
            remaining_tokens=131072,
            token_limit=131072,
            alignment=64,
            allow_minimum=True,
        )
        self.assertEqual(selected, 103104)
        self.assertEqual(
            budget.maximum_aligned_tokens(
                prefix_len=0,
                remaining_tokens=131072,
                token_limit=131072,
                alignment=64,
            ),
            131072,
        )


if __name__ == "__main__":
    unittest.main()
