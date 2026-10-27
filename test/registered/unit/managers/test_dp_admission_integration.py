# SPDX-License-Identifier: Apache-2.0
"""Real IPC, controller and KV-release paths, without loading a GPU model."""

import unittest
import uuid
from array import array
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
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
from sglang.srt.managers.load_snapshot import (
    LoadSnapshot,
    ZmqLoadSnapshotWriter,
    snapshot_decoder,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel, publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
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


class DPAdmissionIPCTestCase(unittest.TestCase):
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


class TestDPAdmissionIPC(DPAdmissionIPCTestCase):
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


class TestDPAdmissionReleaseSnapshots(DPAdmissionIPCTestCase):
    """Real result processing, load inquiry, KV ownership and controller IPC."""

    def scheduler(self, *, capacity=1280, slots=2, enabled=True):
        reset_context()
        args = ServerArgs(
            model_path="dummy",
            speculative_algorithm="EAGLE",
            speculative_num_steps=5,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=6,
        )
        publish(args, role="test")
        self.enterContext(get_parallel().override(dp_rank=0))
        sched = Scheduler.__new__(Scheduler)
        sched.server_args = args
        sched.dp_admission = ledger() if enabled else None
        if enabled:
            sched.dp_admission.decode_reserve = 12
        sched.scheduler_stage_metrics = None
        sched.rust_server = None
        sched.ipc_channels = SimpleNamespace(send_to_tokenizer=MagicMock())
        sched.running_batch = SimpleNamespace(reqs=[])
        sched.waiting_queue = []
        sched.chunked_req = None
        sched.disaggregation_mode = DisaggregationMode.DECODE
        sched.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[]
        )
        sched.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        sched.max_total_num_tokens = capacity
        sched.max_running_requests = slots
        sched.kv_shard_widening = 1
        sched.spec_algorithm = SpeculativeAlgorithm.EAGLE
        sched.tp_worker = None

        # The allocator's grouped free is deliberately observable: dropping a
        # Req's ownership before free_group_end must not refund occupied pages.
        self.used_tokens = 0
        self.pending_free_tokens = 0
        self.in_free_group = False
        self.defer_allocator_free = False
        self.before_free_group_end = lambda: None
        allocator = MagicMock(page_size=64)
        allocator.available_size.side_effect = lambda: capacity - self.used_tokens

        def begin():
            self.in_free_group = True

        def end():
            self.before_free_group_end()
            self.in_free_group = False
            if not self.defer_allocator_free:
                self.used_tokens -= self.pending_free_tokens
                self.pending_free_tokens = 0

        def free_row(kv, spans):
            tokens = (kv.kv_allocated_len + 63) // 64 * 64
            if self.in_free_group:
                self.pending_free_tokens += tokens
            else:
                self.used_tokens -= tokens

        allocator.free_group_begin.side_effect = begin
        allocator.free_group_end.side_effect = end
        sched.token_to_kv_pool_allocator = allocator
        pool = ReqToTokenPool.__new__(ReqToTokenPool)
        pool.free_slots = []
        pool._aux_cache = None
        sched.req_to_token_pool = pool
        tree = MagicMock()
        tree.claim_kv_row.return_value = False
        tree.supports_mamba.return_value = False
        tree.evictable_size.return_value = 0
        tree.free_kv_row.side_effect = free_row
        tree.req_to_token_pool = pool
        tree.token_to_kv_pool_allocator = allocator
        sched.tree_cache = tree
        sched.pool_stats_observer = SchedulerPoolStatsObserver(
            tree_cache=tree,
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=pool,
            session_controller=None,
            hisparse_coordinator=None,
            is_hybrid_swa=False,
            is_hybrid_ssm=False,
            enable_hisparse=False,
            full_tokens_per_layer=None,
            swa_tokens_per_layer=None,
            max_total_num_tokens=capacity,
        )
        sched.metrics_reporter = MagicMock()
        sched.metrics_reporter.spec_total_num_forward_ct = 0
        sched.metrics_reporter.forward_ct_decode = 0
        sched.metrics_reporter.stats = SimpleNamespace(
            gen_throughput=0,
            cache_hit_rate=0,
            utilization=0,
            kv_transfer_speed_gb_s=0,
            kv_transfer_latency_ms=0,
            num_grammar_queue_reqs=0,
            num_paused_reqs=0,
            num_retracted_reqs=0,
        )
        sched.init_load_inquirer()
        self.published = []
        sched.load_snapshot_writer = SimpleNamespace(
            publish_counter=0,
            publish_interval=15,
            write=MagicMock(side_effect=self.published.append),
        )
        sched.load_publisher = MagicMock()
        sched._record_step_counters = MagicMock()
        sched.enable_fpm = False
        sched._maybe_clear_mm_inputs = MagicMock()
        sched.maybe_send_health_check_signal = MagicMock()
        streamer = MagicMock()

        def stream_output(reqs, return_logprob):
            for req in reqs:
                if req.finished():
                    req.finished_output = True

        streamer.stream_output.side_effect = stream_output
        sched.batch_result_processor = SchedulerBatchResultProcessor(
            is_generation=True,
            disaggregation_mode=DisaggregationMode.DECODE,
            enable_overlap=True,
            enable_overlap_mlx=False,
            model_config=SimpleNamespace(self_opened_think_start_ids=None),
            token_to_kv_pool_allocator=allocator,
            tree_cache=tree,
            hisparse_coordinator=None,
            req_to_token_pool=pool,
            decode_offload_manager=None,
            metrics_collector=MagicMock(),
            metrics_reporter=sched.metrics_reporter,
            draft_worker=None,
            model_worker=MagicMock(),
            logprob_result_processor=None,
            output_streamer=streamer,
            beam_coordinator=None,
            abort_request=MagicMock(),
        )
        return sched

    def add_request(self, sched, msg):
        if sched.dp_admission is not None:
            sched.dp_admission.acknowledge(msg)
        msg.sampling_params.normalize(None)
        req = sched._create_generation_req(
            msg, msg.rid, "", msg.input_ids, msg.sampling_params
        )
        req.kv.req_pool_idx = len(sched.running_batch.reqs)
        req.kv.kv_committed_len = len(msg.input_ids)
        req.kv.kv_allocated_len = 128
        self.used_tokens += 128
        sched.running_batch.reqs.append(req)
        return req

    def process(self, sched, reqs=None):
        if reqs is None:
            reqs = sched.running_batch.reqs
        batch = SimpleNamespace(
            reqs=reqs,
            forward_mode=ForwardMode.DECODE,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            batch_size=lambda: len(reqs),
            return_logprob=False,
            mamba_track_mask_cpu=None,
            has_grammar=False,
        )
        result = GenerationBatchResult(
            next_token_ids=torch.tensor([7] * (6 * len(reqs))),
            accept_lens=torch.tensor([6] * len(reqs)),
            speculative_num_draft_tokens=6,
        )
        sched.process_batch_result(batch, result)

    def full_scheduler_and_queue(self):
        ctl = self.controller(queue_size=64)
        sched = self.scheduler()
        sched.publish_load_snapshot(force=True)
        ctl.load_snapshot_reader.read_all.side_effect = lambda: [self.published[-1]]
        for rid, output in (("finishing", 6), ("continuing", 100)):
            ctl._request_dispatcher(request(rid, output=output))
            self.add_request(sched, self.recv(self.receivers[0]))
        sched.publish_load_snapshot(force=True)
        self.now = 1.1
        ctl._request_dispatcher(request("queued", output=100))
        self.assertEqual(len(ctl.dp_admission.queue), 1)
        self.published.clear()
        # Keep the last published snapshot accessible while the next iteration
        # has not produced a fresh one yet.
        last = sched.load_snapshot_writer.write.call_args.args[0]
        ctl.load_snapshot_reader.read_all.side_effect = lambda: [
            self.published[-1] if self.published else last
        ]
        return sched, ctl

    def test_eagle_terminal_batch_counts_only_emitted_reasoning(self):
        sched = self.scheduler()
        req = self.add_request(
            sched,
            request(
                output=2,
                dp_admission_epoch="test",
                dp_admission_sequence=1,
                dp_admission_tokens=640,
            ),
        )
        req.require_reasoning = True
        sched.batch_result_processor.model_config.think_end_ids = [99]
        self.process(sched)
        self.assertEqual(len(req.output_ids), 6)
        self.assertEqual(req.finished_len, 2)
        self.assertEqual(req.reasoning_tokens, 2)
        self.assertEqual(
            sched.dp_admission.snapshot(0)["dp_admission_reserved_requests"], 0
        )

    def test_eagle_completion_publishes_after_group_free_and_wakes_queue(self):
        sched, ctl = self.full_scheduler_and_queue()

        def before_free():
            # A separate /v1/loads read retires the ownership entry before the
            # writer gets a turn, but its release notification must survive.
            observed = sched.load_inquirer.get_loads()
            self.assertEqual(observed.dp_admission_reserved_requests, 1)
            self.assertEqual(observed.dp_admission_reserved_tokens, 768)
            self.assertEqual(observed.num_running_reqs, 1)
            self.assertEqual(len(self.published), 0)
            self.assertEqual(self.used_tokens, 256)

        self.before_free_group_end = before_free
        self.process(sched)
        self.assertEqual(len(self.published), 1)
        load = self.published[-1]
        self.assertEqual(load.dp_admission_reserved_tokens, 640)
        self.assertEqual(load.dp_admission_reserved_requests, 1)
        self.assertEqual(load.num_running_reqs, 1)
        self.assertEqual(load.dp_admission_sequence, 2)
        self.assertEqual(load.dp_admission_epoch, "test")
        self.assertEqual(len(sched.running_batch.reqs), 2)  # Not filtered yet.
        self.now = 1.2
        ctl.poll_dp_admission()
        sent = self.recv(self.receivers[0])
        self.assertEqual((sent.rid, sent.dp_admission_sequence), ("queued", 3))
        self.assertEqual(ctl.dp_admission.ranks[0].total_requests, 2)
        self.assertEqual(len(ctl.dp_admission.queue), 0)

    def test_allocator_floor_drop_publishes_without_another_completion(self):
        sched, ctl = self.full_scheduler_and_queue()
        self.defer_allocator_free = True
        self.process(sched)
        self.assertEqual(self.published[-1].dp_admission_reserved_tokens, 768)
        self.now = 1.2
        ctl.poll_dp_admission()
        self.assertEqual(len(ctl.dp_admission.queue), 1)
        with self.assertRaises(zmq.Again):
            self.recv(self.receivers[0])
        self.used_tokens -= self.pending_free_tokens
        self.pending_free_tokens = 0
        self.defer_allocator_free = False
        self.process(sched, [sched.running_batch.reqs[1]])
        self.assertEqual(len(self.published), 2)
        self.assertEqual(self.published[-1].dp_admission_reserved_tokens, 640)
        self.now = 1.3
        ctl.poll_dp_admission()
        self.assertEqual(self.recv(self.receivers[0]).rid, "queued")

    def test_multiple_completions_coalesce_and_ordinary_decode_stays_throttled(self):
        sched = self.scheduler(capacity=4096, slots=4)
        for sequence, output in enumerate((6, 6, 1000), 1):
            self.add_request(
                sched,
                request(
                    str(sequence),
                    output=output,
                    dp_admission_epoch="test",
                    dp_admission_sequence=sequence,
                    dp_admission_tokens=640 if output == 6 else 1152,
                ),
            )
        sched.publish_load_snapshot(force=True)
        self.published.clear()
        self.process(sched)
        self.assertEqual(len(self.published), 1)
        self.assertEqual(sched.dp_admission.num_released_requests, 2)
        self.assertEqual(self.published[-1].dp_admission_reserved_requests, 1)
        survivor = sched.running_batch.reqs[2]
        for _ in range(14):
            # Growing live KV is covered by its existing commitment and must
            # not turn the release notification into per-token publication.
            allocated = (survivor.kv.kv_committed_len + 6 + 63) // 64 * 64
            self.used_tokens += allocated - survivor.kv.kv_allocated_len
            survivor.kv.kv_allocated_len = allocated
            self.process(sched, [survivor])
        self.assertEqual(len(self.published), 1)
        self.process(sched, [survivor])
        self.assertEqual(len(self.published), 2)

    def test_failed_release_publication_retries_without_losing_notification(self):
        sched, _ = self.full_scheduler_and_queue()
        writer = sched.load_snapshot_writer
        writer.write.side_effect = OSError("temporarily unavailable")
        self.process(sched)
        self.assertEqual(len(self.published), 0)
        writer.write.side_effect = self.published.append
        self.process(sched, [sched.running_batch.reqs[1]])
        self.assertEqual(len(self.published), 1)
        self.assertEqual(self.published[-1].dp_admission_reserved_requests, 1)

    def test_new_binding_cannot_hide_an_equal_cost_completion(self):
        sched = self.scheduler()
        for sequence, output in ((1, 6), (2, 100)):
            self.add_request(
                sched,
                request(
                    str(sequence),
                    output=output,
                    dp_admission_epoch="test",
                    dp_admission_sequence=sequence,
                    dp_admission_tokens=640,
                ),
            )
            if sequence == 1:
                sched.publish_load_snapshot(force=True)
        self.published.clear()
        # One newly acknowledged commitment replaces one completed commitment.
        # The aggregate tokens/slots match the old snapshot, but that snapshot
        # plus the controller's unacknowledged dispatch still overstates load.
        self.process(sched)
        self.assertEqual(len(self.published), 1)
        self.assertEqual(self.published[-1].dp_admission_reserved_tokens, 640)
        self.assertEqual(self.published[-1].dp_admission_reserved_requests, 1)
        self.assertEqual(self.published[-1].dp_admission_sequence, 2)

    def test_zmq_backpressure_does_not_acknowledge_release_publication(self):
        sched, _ = self.full_scheduler_and_queue()
        writer = ZmqLoadSnapshotWriter.__new__(ZmqLoadSnapshotWriter)
        writer.dp_rank = 0
        writer.publish_counter = 0
        writer.publish_interval = 15
        writer._zmq = zmq
        writer._socket = MagicMock()
        writer._socket.send.side_effect = zmq.Again()
        sched.load_snapshot_writer = writer
        self.process(sched)
        self.assertEqual(writer._socket.send.call_count, 1)
        self.assertEqual(sched._last_published_dp_admission, (0, 1280))
        writer._socket.send.side_effect = None
        self.process(sched, [sched.running_batch.reqs[1]])
        self.assertEqual(writer._socket.send.call_count, 2)
        load = snapshot_decoder.decode(writer._socket.send.call_args.args[0])
        self.assertEqual(load.dp_admission_sequence, 2)
        self.assertEqual(load.dp_admission_reserved_tokens, 640)
        self.assertEqual(sched._last_published_dp_admission, (1, 640))

    def test_mooncake_deferred_ownership_is_not_refunded_until_drain(self):
        sched, ctl = self.full_scheduler_and_queue()
        held, continuing = sched.running_batch.reqs
        held.finished_output = True
        held.bootstrap_room = 42
        # Even a stale running-list reference cannot hide held KV ownership.
        self.assertEqual(sched.load_inquirer.get_loads().num_running_reqs, 2)
        sched.running_batch.reqs = [continuing]
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.scheduler = sched
        sched.enable_hisparse = False
        sched.metrics_reporter.enable_metrics = False
        queue.queue = []
        queue.tree_cache = sched.tree_cache
        queue.enable_staging = queue.enable_host_receive = False
        queue.metadata_buffers = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.gloo_group = None
        receiver = MagicMock()
        receiver.kv_mgr.is_abort_release_safe.return_value = False
        decode_req = DecodeRequest(held, receiver)
        queue._deferred_releases = [(decode_req, 0.0, float("inf"), 3, 1)]
        sched.disagg_decode_transfer_queue = queue
        queue.resolve_deferred_releases()
        self.process(sched)
        self.assertTrue(held.kv.holds_kv)
        self.assertEqual(len(self.published), 0)
        self.assertEqual(sched.dp_admission.num_released_requests, 0)
        self.now = 1.2
        ctl.poll_dp_admission()
        self.assertEqual(len(ctl.dp_admission.queue), 1)
        with self.assertRaises(zmq.Again):
            self.recv(self.receivers[0])
        receiver.kv_mgr.is_abort_release_safe.return_value = True
        queue.resolve_deferred_releases()
        self.assertFalse(held.kv.holds_kv)
        self.assertIsNone(decode_req.kv_receiver)
        self.process(sched)
        self.assertEqual(len(self.published), 1)
        self.assertEqual(self.published[-1].dp_admission_reserved_tokens, 640)
        self.now = 1.3
        ctl.poll_dp_admission()
        self.assertEqual(self.recv(self.receivers[0]).rid, "queued")

    def test_default_off_preserves_snapshot_timing_and_public_fields(self):
        sched = self.scheduler(enabled=False)
        req = self.add_request(sched, request(output=6))
        sched.load_snapshot_writer.publish_counter = 14
        self.process(sched)
        self.assertEqual(len(self.published), 1)
        self.assertEqual(self.published[0].num_running_reqs, 1)
        self.assertEqual(self.published[0].num_used_tokens, 128)
        self.assertFalse(req.kv.holds_kv)
        self.assertFalse(
            any(key.startswith("dp_admission_") for key in self.published[0].to_dict())
        )
        # Even a direct read retains the pre-existing raw running-list count.
        self.assertEqual(sched.load_inquirer.get_loads().num_running_reqs, 1)


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
