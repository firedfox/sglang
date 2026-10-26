# SPDX-License-Identifier: Apache-2.0
"""Real IPC, controller and KV-release paths, without loading a GPU model."""

import unittest
import uuid
from array import array
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import zmq

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.decode import DecodeRequest, DecodeTransferQueue
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.data_parallel_controller import (
    DataParallelController,
    DPBudget,
    LoadBalanceMethod,
)
from sglang.srt.managers.dp_admission import (
    DPAdmissionController,
    SchedulerDPAdmission,
)
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchTokenizedGenerateReqInput,
    TokenizedGenerateReqInput,
    sock_recv,
)
from sglang.srt.managers.load_snapshot import LoadSnapshot
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def request(rid="user", prompt=100, output=20, **fields):
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text=None,
        input_ids=array("q", [1] * prompt),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=output),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        **fields,
    )


def snapshot(rank=0, timestamp=1.0, **fields):
    values = dict(
        timestamp=timestamp,
        dp_rank=rank,
        max_total_num_tokens=4096,
        max_running_requests=1,
        dp_admission_epoch="test",
        dp_admission_page_size=64,
        dp_admission_decode_reserve=8,
        dp_admission_prealloc_reserve=512,
        dp_admission_max_seq_len=4096,
    )
    values.update(fields)
    return LoadSnapshot(**values)


def ledger():
    return SchedulerDPAdmission(
        "test",
        page_size=64,
        decode_reserve=8,
        prealloc_reserve=512,
        max_sequence_length=4096,
    )


class TestDPAdmissionIPC(unittest.TestCase):
    def setUp(self):
        reset_context()
        publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(reset_context)
        self.context = zmq.Context()
        self.addCleanup(self.context.term)
        self.sockets = []
        self.addCleanup(lambda: [sock.close(linger=0) for sock in self.sockets])
        self.now = 1.0
        clock = patch(
            "sglang.srt.managers.data_parallel_controller.time.perf_counter",
            side_effect=lambda: self.now,
        )
        clock.start()
        self.addCleanup(clock.stop)

    def pair(self):
        sender, receiver = self.context.socket(zmq.PUSH), self.context.socket(zmq.PULL)
        self.sockets.extend((sender, receiver))
        endpoint = f"inproc://dp-admission-{uuid.uuid4().hex}"
        receiver.bind(endpoint)
        sender.connect(endpoint)
        return sender, receiver

    def controller(self, ranks=1, queue_size=0):
        ctl = DataParallelController.__new__(DataParallelController)
        pairs = [self.pair() for _ in range(ranks)]
        ctl.workers = [pair[0] for pair in pairs]
        self.receivers = [pair[1] for pair in pairs]
        ctl.status = [True] * ranks
        ctl._active_workers = list(range(ranks))
        ctl.control_message_step = 1
        ctl.load_balance_method = LoadBalanceMethod.TOTAL_TOKENS
        ctl.refresh_load_budget_on_dispatch = True
        ctl.dp_budget = DPBudget(ranks)
        ctl.dp_admission = DPAdmissionController(
            ranks, "test", max_queued_requests=queue_size, queue_timeout=1
        )
        ctl._last_refresh_time = 0
        ctl.load_snapshot_reader = MagicMock()
        ctl.load_snapshot_reader.read_all.return_value = [
            snapshot(i) for i in range(ranks)
        ]
        output, self.rejections = self.pair()
        ctl.send_to_tokenizer = SenderWrapper(output)
        ctl.init_dispatcher()
        return ctl

    def recv(self, socket):
        return sock_recv(socket, flags=zmq.NOBLOCK)

    def test_dispatch_encodes_reservation_and_preserves_sampling(self):
        ctl = self.controller()
        req = request(output=37, http_worker_ipc="worker-7")
        ctl._request_dispatcher(req)
        sent = self.recv(self.receivers[0])
        self.assertEqual(sent.dp_admission_sequence, 1)
        self.assertEqual(sent.dp_admission_tokens, 640)
        self.assertEqual(sent.dp_admission_epoch, "test")
        self.assertEqual(sent.sampling_params.max_new_tokens, 37)
        self.assertEqual(sent.http_worker_ipc, "worker-7")
        self.assertIsNotNone(sent.time_stats)

    def test_full_budget_returns_routed_abort_without_worker_dispatch(self):
        ctl = self.controller()
        ctl._request_dispatcher(request("one"))
        self.recv(self.receivers[0])
        ctl._request_dispatcher(request("two", http_worker_ipc="tokenizer-2"))
        rejected = self.recv(self.rejections)
        self.assertEqual(rejected.rid, "two")
        self.assertEqual(rejected.http_worker_ipc, "tokenizer-2")
        self.assertEqual(rejected.finished_reason["status_code"], 429)
        self.assertEqual(rejected.finished_reason["dp_admission_rejected"], 1)
        with self.assertRaises(zmq.Again):
            self.recv(self.receivers[0])

    def test_unknown_and_impossible_requests_have_distinct_status(self):
        for loads, prompt, expected in (([], 1, 503), ([snapshot()], 4096, 400)):
            with self.subTest(status=expected):
                ctl = self.controller()
                ctl.load_snapshot_reader.read_all.return_value = loads
                ctl._request_dispatcher(request(prompt=prompt))
                rejected = self.recv(self.rejections)
                self.assertEqual(rejected.finished_reason["status_code"], expected)
                self.assertEqual(len(ctl.dp_admission.queue), 0)

    def test_snapshot_read_failure_returns_503_without_dispatch(self):
        ctl = self.controller()
        ctl.load_snapshot_reader.read_all.side_effect = OSError("snapshot unavailable")
        ctl._request_dispatcher(request())
        self.assertEqual(self.recv(self.rejections).finished_reason["status_code"], 503)
        self.assertEqual(ctl.dp_admission.ranks[0].total_requests, 0)
        with self.assertRaises(zmq.Again):
            self.recv(self.receivers[0])

    def test_external_rank_budget_is_enforced_despite_free_other_rank(self):
        ctl = self.controller(ranks=2)
        ctl._request_dispatcher(request("one", routed_dp_rank=0))
        ctl._request_dispatcher(request("two", routed_dp_rank=0))
        self.assertEqual(self.recv(self.rejections).finished_reason["status_code"], 429)
        with self.assertRaises(zmq.Again):
            self.recv(self.receivers[1])

    def test_queued_request_drains_without_new_inbound_request(self):
        ctl = self.controller(queue_size=1)
        ctl._request_dispatcher(request("one"))
        ctl._request_dispatcher(request("two"))
        self.recv(self.receivers[0])
        self.assertEqual(len(ctl.dp_admission.queue), 1)
        self.now = 1.1
        ctl.load_snapshot_reader.read_all.return_value = [
            snapshot(timestamp=2, dp_admission_sequence=1)
        ]
        ctl.poll_dp_admission()
        sent = self.recv(self.receivers[0])
        self.assertEqual((sent.rid, sent.dp_admission_sequence), ("two", 2))
        self.assertIsNotNone(sent.time_stats)  # Queued object still encodes correctly.
        self.assertEqual(len(ctl.dp_admission.queue), 0)

    def test_queue_timeout_and_limit_do_not_leak_reservations(self):
        ctl = self.controller(queue_size=1)
        for rid in ("active", "queued", "overflow"):
            ctl._request_dispatcher(request(rid))
        self.assertEqual(self.recv(self.rejections).rid, "overflow")
        self.now = 2.1
        ctl.poll_dp_admission()
        expired = self.recv(self.rejections)
        self.assertEqual(expired.rid, "queued")
        self.assertEqual(expired.finished_reason["status_code"], 429)
        self.assertEqual(len(ctl.dp_admission.queue), 0)
        self.assertEqual(ctl.dp_admission.ranks[0].total_requests, 1)

    def test_abort_clears_controller_queue_but_not_sent_commitment(self):
        ctl = self.controller(queue_size=1)
        ctl._request_dispatcher(request("batch-0"))
        ctl._request_dispatcher(request("batch-1"))
        self.recv(self.receivers[0])
        ctl._request_dispatcher(AbortReq(rid="batch"))
        self.assertEqual(self.recv(self.rejections).rid, "batch-1")
        self.assertIsInstance(self.recv(self.receivers[0]), AbortReq)
        self.assertEqual(len(ctl.dp_admission.queue), 0)
        self.assertEqual(ctl.dp_admission.ranks[0].total_requests, 1)

    def test_failed_send_rolls_back_and_batch_continues(self):
        ctl = self.controller(ranks=2)
        ctl.workers[0].close(linger=0)
        batch = BatchTokenizedGenerateReqInput(
            batch=[
                request("failed", routed_dp_rank=0),
                request("sent", routed_dp_rank=1),
                request("full", routed_dp_rank=1),
            ]
        )
        ctl._request_dispatcher(batch)
        failed, full = self.recv(self.rejections), self.recv(self.rejections)
        self.assertEqual(
            (failed.rid, failed.finished_reason["status_code"]), ("failed", 503)
        )
        self.assertEqual((full.rid, full.finished_reason["status_code"]), ("full", 429))
        self.assertEqual(self.recv(self.receivers[1]).rid, "sent")
        self.assertEqual(ctl.dp_admission.ranks[0].total_requests, 0)
        self.assertEqual(ctl.dp_admission.ranks[1].total_requests, 1)

    def test_trace_preparation_failure_refunds_unsent_reservation(self):
        ctl = self.controller()
        with patch(
            "sglang.srt.managers.data_parallel_controller.DPControllerReqTimeStats.new_from_obj",
            side_effect=ValueError("invalid trace"),
        ):
            ctl._request_dispatcher(request())
        self.assertEqual(self.recv(self.rejections).finished_reason["status_code"], 503)
        self.assertEqual(ctl.dp_admission.ranks[0].total_requests, 0)
        with self.assertRaises(zmq.Again):
            self.recv(self.receivers[0])

    def test_unsupported_session_rejected_before_allocation(self):
        ctl = self.controller()
        ctl._request_dispatcher(request(session_id="retained-kv"))
        self.assertEqual(self.recv(self.rejections).finished_reason["status_code"], 400)
        self.assertEqual(ctl.dp_admission.ranks[0].total_requests, 0)

    def test_health_prefix_cannot_bypass_stalled_transfer_capacity(self):
        ctl = self.controller()
        ctl.load_snapshot_reader.read_all.return_value = [
            snapshot(
                dp_admission_reserved_requests=1,
                dp_admission_reserved_tokens=640,
                num_waiting_reqs=1,
            )
        ]
        # Real health-idle semantics deliberately ignore a stalled transfer.
        sched = Scheduler.__new__(Scheduler)
        sched.running_batch = SimpleNamespace(is_empty=lambda: True)
        sched.chunked_req = None
        sched.dllm_manager = SimpleNamespace(any_staging_reqs=lambda: False)
        sched.last_batch = None
        sched.enable_overlap = False
        sched.waiting_queue = []
        sched._engine_paused = False
        sched.disaggregation_mode = DisaggregationMode.DECODE
        sched.disagg_decode_prealloc_queue = SimpleNamespace(retracted_queue=[])
        sched.disagg_decode_transfer_queue = SimpleNamespace(queue=[object()])
        self.assertTrue(sched.is_fully_idle(for_health_check=True))
        ctl._request_dispatcher(request(f"{HEALTH_CHECK_RID_PREFIX}_client-controlled"))
        self.assertEqual(self.recv(self.rejections).finished_reason["status_code"], 429)
        self.assertEqual(ctl.dp_admission.ranks[0].total_requests, 1)
        with self.assertRaises(zmq.Again):
            self.recv(self.receivers[0])


class TestSchedulerAdmissionRelease(unittest.TestCase):
    def setUp(self):
        reset_context()
        publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(reset_context)

    def scheduler(self):
        sched = Scheduler.__new__(Scheduler)
        sched.dp_admission = ledger()
        sched.rust_server = None
        sched.ipc_channels = SimpleNamespace(send_to_tokenizer=MagicMock())
        return sched

    def bind(self, sched):
        msg = request()
        msg.dp_admission_epoch = "test"
        msg.dp_admission_sequence = 1
        msg.dp_admission_tokens = 640
        sched.dp_admission.acknowledge(msg)
        req = sched._create_generation_req(
            msg, msg.rid, "", msg.input_ids, msg.sampling_params
        )
        return req

    def test_actual_mooncake_drain_and_kv_release_are_required_for_refund(self):
        sched = self.scheduler()
        req = self.bind(sched)
        req.kv.req_pool_idx = 1
        req.kv.kv_allocated_len = req.kv.kv_committed_len = 128
        req.finished_output = True
        req.bootstrap_room = 42
        pool = ReqToTokenPool.__new__(ReqToTokenPool)
        pool.free_slots = []
        pool._aux_cache = None
        tree = MagicMock()
        tree.claim_kv_row.return_value = False
        tree.req_to_token_pool = pool
        tree.token_to_kv_pool_allocator.page_size = 64
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.scheduler = SimpleNamespace(
            enable_hisparse=False,
            metrics_reporter=SimpleNamespace(enable_metrics=False),
        )
        queue.tree_cache = tree
        queue.enable_staging = queue.enable_host_receive = False
        queue.metadata_buffers = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.gloo_group = None
        receiver = MagicMock()
        receiver.kv_mgr.is_abort_release_safe.return_value = False
        decode_req = DecodeRequest(req, receiver)
        queue._deferred_releases = [(decode_req, 0.0, float("inf"), 3, 1)]
        with patch(
            "sglang.srt.disaggregation.decode.all_reduce_flags",
            side_effect=lambda flags, **kwargs: flags,
        ):
            queue.resolve_deferred_releases()
            self.assertEqual(
                sched.dp_admission.snapshot(128)["dp_admission_reserved_requests"], 1
            )
            self.assertTrue(req.kv.holds_kv)
            self.assertEqual(pool.free_slots, [])
            receiver.kv_mgr.is_abort_release_safe.return_value = True
            queue.resolve_deferred_releases()
        self.assertFalse(req.kv.holds_kv)
        self.assertEqual(pool.free_slots, [1])
        tree.free_kv_row.assert_called_once()
        self.assertEqual(
            sched.dp_admission.snapshot(0)["dp_admission_reserved_requests"], 0
        )
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        self.assertIsNone(decode_req.kv_receiver)

    def test_abort_output_marks_terminal_but_waits_for_release(self):
        sched = self.scheduler()
        req = self.bind(sched)
        req.kv.req_pool_idx = 1
        req.kv.kv_allocated_len = 128
        sched.send_abort_output(AbortReq(rid=req.rid), req)
        self.assertEqual(
            sched.dp_admission.snapshot(128)["dp_admission_reserved_requests"], 1
        )
        req.kv.req_pool_idx = None
        req.kv.mark_kv_released()
        self.assertEqual(
            sched.dp_admission.snapshot(0)["dp_admission_reserved_requests"], 0
        )

    def test_busy_health_skip_acknowledges_without_creating_req(self):
        sched = self.scheduler()
        sched.scheduler_stage_metrics = None
        sched.session_controller = MagicMock()
        sched.is_fully_idle = MagicMock(return_value=False)
        sched.return_health_check_ipcs = deque()
        sched.flush_wrapper = MagicMock()
        sched.external_corpus_manager = None
        sched._request_dispatcher = MagicMock()
        msg = request(f"{HEALTH_CHECK_RID_PREFIX}_probe")
        msg.dp_admission_epoch = "test"
        msg.dp_admission_sequence = 1
        msg.dp_admission_tokens = 576
        sched.process_input_requests([msg])
        load = sched.dp_admission.snapshot(0)
        self.assertEqual(load["dp_admission_sequence"], 1)
        self.assertEqual(load["dp_admission_reserved_requests"], 0)
        sched._request_dispatcher.assert_not_called()


if __name__ == "__main__":
    unittest.main()
