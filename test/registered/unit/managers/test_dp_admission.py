# SPDX-License-Identifier: Apache-2.0
"""CPU tests for request commitments, ACK races, and actual KV ownership."""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.dp_admission import (
    DPAdmissionController,
    DPAdmissionError,
    DPAdmissionRequest,
    SchedulerDPAdmission,
)
from sglang.srt.managers.load_snapshot import (
    LoadSnapshot,
    snapshot_decoder,
    snapshot_encoder,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def snapshot(rank=0, timestamp=1.0, **fields):
    values = dict(
        dp_rank=rank,
        timestamp=timestamp,
        max_total_num_tokens=4096,
        max_running_requests=2,
        dp_admission_epoch="test",
        dp_admission_page_size=64,
        dp_admission_decode_reserve=8,
        dp_admission_prealloc_reserve=512,
        dp_admission_max_seq_len=4096,
    )
    values.update(fields)
    return LoadSnapshot(**values)


def controller(ranks=1, **kwargs):
    policy = DPAdmissionController(ranks, "test", **kwargs)
    policy.update([snapshot(i) for i in range(ranks)], now=1.0)
    return policy


def reserve(policy, request=None, now=1.0):
    choice = policy.choose(
        request or DPAdmissionRequest(100, 20),
        list(range(len(policy.ranks))),
        now,
    )
    return policy.reserve(*choice)


def live_request(*, rid="same-rid", allocated=128, finished=False, backup=None):
    return SimpleNamespace(
        rid=rid,
        finished_output=finished,
        kv=SimpleNamespace(
            holds_kv=allocated > 0,
            holds_mamba=False,
            retraction_backup=backup,
            kv_allocated_len=allocated,
        ),
    )


def scheduler_ledger():
    return SchedulerDPAdmission(
        "test",
        page_size=64,
        decode_reserve=8,
        prealloc_reserve=512,
        max_sequence_length=4096,
    )


def bind(ledger, req, sequence=1, tokens=640):
    msg = SimpleNamespace(
        dp_admission_epoch="test",
        dp_admission_sequence=sequence,
        dp_admission_tokens=tokens,
    )
    ledger.acknowledge(msg)
    ledger.bind(req, msg)


class TestDPAdmissionController(unittest.TestCase):
    def assert_status(self, expected, call):
        with self.assertRaises(DPAdmissionError) as caught:
            call()
        self.assertEqual(caught.exception.status_code, expected)

    def test_snapshot_protocol_roundtrip_and_default_off_public_schema(self):
        load = snapshot(
            dp_admission_sequence=7,
            dp_admission_reserved_tokens=4096,
            dp_admission_reserved_requests=2,
        )
        decoded = snapshot_decoder.decode(snapshot_encoder.encode(load))
        self.assertEqual(decoded, load)
        self.assertEqual(decoded.to_dict({"core"})["dp_admission_sequence"], 7)
        self.assertFalse(
            any(key.startswith("dp_admission_") for key in LoadSnapshot().to_dict())
        )

    def test_prompt_generation_double_buffer_and_prealloc_floor(self):
        policy = controller()
        short = policy.choose(DPAdmissionRequest(100, 1), [0], 1.0)
        long = policy.choose(DPAdmissionRequest(100, 1000), [0], 1.0)
        self.assertEqual(short, (0, 640))  # ceil((100 + 512) / 64)
        self.assertEqual(long, (0, 1152))  # ceil((100 + 1000 + 8) / 64)

    def test_context_and_server_generation_limits_without_request_mutation(self):
        policy = controller()
        policy.update(
            [snapshot(timestamp=2, dp_admission_max_new_tokens_limit=32)], now=2
        )
        request = DPAdmissionRequest(100, None)
        self.assertEqual(policy.choose(request, [0], 2), (0, 640))
        self.assertIsNone(request.max_new_tokens)
        policy.update(
            [
                snapshot(
                    timestamp=3,
                    dp_admission_max_seq_len=1024,
                    dp_admission_prealloc_reserve=0,
                )
            ],
            now=3,
        )
        self.assertEqual(policy.choose(request, [0], 3), (0, 1088))

    def test_non_page_aligned_prealloc_reserve_cannot_undercharge(self):
        policy = controller()
        policy.update(
            [
                snapshot(
                    timestamp=2,
                    max_total_num_tokens=640,
                    dp_admission_prealloc_reserve=513,
                )
            ],
            now=2,
        )
        # The allocator reserves ceil(100/64)*64 + 513 = 641, not 613.
        self.assert_status(
            400, lambda: policy.choose(DPAdmissionRequest(100, 1), [0], 2)
        )

    def test_slot_limit_counts_all_unacknowledged_dispatches(self):
        policy = controller()
        reserve(policy)
        reserve(policy)
        self.assert_status(429, lambda: reserve(policy))

    def test_generation_commitment_limits_tokens_before_slots(self):
        policy = controller()
        policy.update(
            [snapshot(timestamp=2, max_total_num_tokens=1024, max_running_requests=32)],
            now=2,
        )
        reserve(policy, now=2)
        self.assert_status(429, lambda: reserve(policy, now=2))

    def test_new_timestamp_without_ack_does_not_erase_inflight_reservations(self):
        policy = controller()
        first = reserve(policy)
        policy.update([snapshot(timestamp=2)], now=2)
        second = reserve(policy, now=2)
        self.assertEqual((first.sequence, second.sequence), (1, 2))
        self.assertEqual(policy.ranks[0].total_requests, 2)
        self.assert_status(429, lambda: reserve(policy, now=2))

    def test_ack_replaces_only_acknowledged_reservations_once(self):
        policy = controller()
        first, second = reserve(policy), reserve(policy)
        policy.update(
            [
                snapshot(
                    timestamp=2,
                    dp_admission_sequence=first.sequence,
                    dp_admission_reserved_requests=1,
                    dp_admission_reserved_tokens=first.tokens,
                    num_total_tokens=128,
                )
            ],
            now=2,
        )
        self.assertEqual(policy.ranks[0].pending, {second.sequence: second.tokens})
        self.assertEqual(policy.ranks[0].total_requests, 2)
        self.assertEqual(policy.ranks[0].total_tokens, first.tokens + second.tokens)
        policy.update([snapshot(timestamp=3, dp_admission_sequence=2)], now=3)
        self.assertEqual(policy.ranks[0].total_requests, 0)
        self.assertEqual(policy.ranks[0].total_tokens, 0)

    def test_stale_duplicate_and_regressing_snapshots_do_not_reset_ledger(self):
        policy = controller(snapshot_timeout=5)
        reserve(policy)
        policy.update([snapshot(timestamp=0.5)], now=3)
        policy.update([snapshot(timestamp=1)], now=5)
        self.assertEqual(policy.ranks[0].total_requests, 1)
        self.assert_status(503, lambda: reserve(policy, now=6.1))

    def test_unknown_capacity_and_wrong_epoch_fail_closed(self):
        policy = DPAdmissionController(1, "test")
        self.assert_status(503, lambda: reserve(policy))
        policy.update([snapshot(dp_admission_epoch="old")], now=1)
        self.assert_status(503, lambda: reserve(policy))

    def test_invalid_ack_cannot_clear_pending_reservation(self):
        policy = controller()
        first = reserve(policy)
        policy.update(
            [snapshot(timestamp=2, dp_admission_sequence=first.sequence + 1)], now=2
        )
        self.assertEqual(policy.ranks[0].pending, {1: first.tokens})
        self.assert_status(503, lambda: reserve(policy, now=2))

    def test_impossible_request_is_400_not_a_permanent_wait(self):
        policy = controller()
        too_big = DPAdmissionRequest(4096, 1)
        self.assert_status(400, lambda: reserve(policy, too_big))
        self.assertEqual(len(policy.queue), 0)

    def test_unknown_other_rank_does_not_prove_request_impossible(self):
        policy = DPAdmissionController(2, "test")
        policy.update([snapshot(max_total_num_tokens=64)], now=1)
        self.assert_status(503, lambda: reserve(policy))

    def test_external_rank_cannot_bypass_budget_or_fall_back(self):
        policy = controller(2)
        pinned = DPAdmissionRequest(100, 20, routed_dp_rank=0)
        reserve(policy, pinned)
        reserve(policy, pinned)
        self.assert_status(429, lambda: reserve(policy, pinned))
        self.assertEqual(reserve(policy).rank, 1)
        self.assert_status(
            400,
            lambda: policy.choose(DPAdmissionRequest(1, 1, 8), [0, 1], 1),
        )

    def test_send_failure_rolls_back_only_unsent_and_allows_sequence_gap(self):
        policy = controller()
        failed = reserve(policy)
        policy.rollback_unsent(failed)
        policy.rollback_unsent(failed)
        sent = reserve(policy)
        self.assertEqual(sent.sequence, 2)
        self.assertEqual(policy.ranks[0].total_requests, 1)
        policy.update([snapshot(timestamp=2, dp_admission_sequence=2)], now=2)
        self.assertEqual(policy.ranks[0].total_requests, 0)

    def test_queue_bound_and_prefix_abort_do_not_touch_dispatched_budget(self):
        policy = controller(max_queued_requests=2)
        reserve(policy)
        for rid in ("batch-0", "other"):
            policy.enqueue(SimpleNamespace(rid=rid), DPAdmissionRequest(1, 1), 1)
        self.assert_status(
            429,
            lambda: policy.enqueue(
                SimpleNamespace(rid="overflow"), DPAdmissionRequest(1, 1), 1
            ),
        )
        self.assertEqual([r.rid for r in policy.abort_queued("batch")], ["batch-0"])
        self.assertEqual(len(policy.queue), 1)
        self.assertEqual(policy.abort_queued(""), [])
        self.assertEqual([r.rid for r in policy.abort_queued("", True)], ["other"])
        self.assertEqual(policy.ranks[0].total_requests, 1)

    def test_health_check_is_reserved_and_its_skip_is_acknowledged(self):
        policy = controller()
        reserve(policy)
        choice = policy.choose(DPAdmissionRequest(1, 1), [0], 1)
        health = policy.reserve(*choice)
        self.assertEqual(policy.ranks[0].total_requests, 2)
        policy.update(
            [
                snapshot(
                    timestamp=2,
                    dp_admission_sequence=health.sequence,
                    dp_admission_reserved_requests=1,
                    dp_admission_reserved_tokens=640,
                )
            ],
            now=2,
        )
        self.assertEqual(policy.ranks[0].total_requests, 1)


class TestSchedulerDPAdmission(unittest.TestCase):
    def test_active_and_preallocated_tokens_are_not_counted_twice(self):
        ledger = scheduler_ledger()
        bind(ledger, live_request(), tokens=640)
        load = ledger.snapshot(num_used_tokens=128)
        self.assertEqual(load["dp_admission_reserved_tokens"], 640)
        self.assertEqual(load["dp_admission_reserved_requests"], 1)

    def test_finished_output_does_not_release_transfer_kv(self):
        ledger = scheduler_ledger()
        req = live_request(finished=True)
        bind(ledger, req)
        self.assertEqual(ledger.snapshot(128)["dp_admission_reserved_requests"], 1)
        req.kv.holds_kv = False
        req.kv.kv_allocated_len = 0
        self.assertEqual(ledger.snapshot(0)["dp_admission_reserved_requests"], 0)

    def test_actual_deferred_allocator_free_retains_token_floor(self):
        ledger = scheduler_ledger()
        req = live_request(finished=True)
        bind(ledger, req)
        req.kv.holds_kv = False
        req.kv.kv_allocated_len = 0
        load = ledger.snapshot(128)
        self.assertEqual(load["dp_admission_reserved_requests"], 0)
        self.assertEqual(load["dp_admission_reserved_tokens"], 128)
        self.assertEqual(ledger.snapshot(0)["dp_admission_reserved_tokens"], 0)

    def test_abort_echo_waits_for_kv_release(self):
        ledger = scheduler_ledger()
        req = live_request()
        bind(ledger, req)
        ledger.mark_terminal(req)
        self.assertEqual(ledger.snapshot(128)["dp_admission_reserved_requests"], 1)
        req.kv.holds_kv = False
        req.kv.kv_allocated_len = 0
        self.assertEqual(ledger.snapshot(0)["dp_admission_reserved_requests"], 0)

    def test_retraction_and_resume_keep_one_commitment(self):
        ledger = scheduler_ledger()
        req = live_request(allocated=0, backup=object())
        bind(ledger, req)
        self.assertEqual(ledger.snapshot(0)["dp_admission_reserved_tokens"], 640)
        req.kv.retraction_backup = None
        req.kv.holds_kv = True
        req.kv.kv_allocated_len = 192
        self.assertEqual(ledger.snapshot(192)["dp_admission_reserved_tokens"], 640)
        req.finished_output = True
        req.kv.holds_kv = False
        req.kv.retraction_backup = object()
        self.assertEqual(ledger.snapshot(0)["dp_admission_reserved_requests"], 1)
        req.kv.retraction_backup = None
        self.assertEqual(ledger.snapshot(0)["dp_admission_reserved_requests"], 0)

    def test_same_rid_retry_has_independent_request_identity(self):
        ledger = scheduler_ledger()
        old = live_request(finished=True)
        new = live_request()
        bind(ledger, old, sequence=1)
        bind(ledger, new, sequence=2)
        self.assertEqual(ledger.snapshot(256)["dp_admission_reserved_requests"], 2)
        old.kv.holds_kv = False
        old.kv.kv_allocated_len = 0
        self.assertEqual(ledger.snapshot(128)["dp_admission_reserved_requests"], 1)
        self.assertIs(ledger.live[id(new)].request, new)

    def test_terminal_unallocated_error_and_untracked_allocation(self):
        ledger = scheduler_ledger()
        req = live_request(allocated=0)
        bind(ledger, req)
        ledger.mark_terminal(req)
        load = ledger.snapshot(256)
        self.assertEqual(load["dp_admission_reserved_requests"], 0)
        self.assertEqual(load["dp_admission_reserved_tokens"], 256)

    def test_duplicate_or_wrong_epoch_dispatch_is_rejected(self):
        ledger = scheduler_ledger()
        req = live_request()
        bind(ledger, req)
        for epoch, sequence in (("test", 1), ("old", 2)):
            with self.subTest(epoch=epoch, sequence=sequence):
                with self.assertRaises(RuntimeError):
                    ledger.acknowledge(
                        SimpleNamespace(
                            dp_admission_epoch=epoch,
                            dp_admission_sequence=sequence,
                            dp_admission_tokens=640,
                        )
                    )
        self.assertEqual(ledger.sequence, 1)


if __name__ == "__main__":
    unittest.main()
