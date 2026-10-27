# SPDX-License-Identifier: Apache-2.0
"""Exercise controller construction and its rejection socket without GPU workers."""

import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import Mock, patch

import zmq

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import data_parallel_controller as controller_module
from sglang.srt.managers.data_parallel_controller import (
    DataParallelController,
    LoadBalanceMethod,
)
from sglang.srt.managers.dp_admission import DPAdmissionController, DPAdmissionError
from sglang.srt.managers.io_struct import AbortReq, sock_recv
from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils.network import get_zmq_socket
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDataParallelControllerInit(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        self.context = zmq.Context()
        self.addCleanup(lambda: self.context.destroy(linger=0))

    def construct(self, *, admission, attention_dp=True):
        args = ServerArgs(
            model_path="dummy",
            tp_size=2 if attention_dp else 1,
            attn_dp_size=2 if attention_dp else 1,
            dp_size=1 if attention_dp else 2,
            disaggregation_mode="decode",
            load_balance_method="total_tokens",
            disable_radix_cache=True,
            enable_dp_admission=admission,
            dp_admission_max_queued_requests=17,
            dp_admission_queue_timeout=0.75,
            dp_admission_snapshot_timeout=13.0,
            enable_metrics=False,
        )
        publish(args, role="dp_controller")
        epoch = uuid.uuid4().hex
        ports = PortArgs(
            tokenizer_ipc_name=f"inproc://controller-init-output-{epoch}",
            scheduler_input_ipc_name=f"inproc://controller-init-input-{epoch}",
            detokenizer_ipc_name="unused",
            nccl_port=0,
            rpc_ipc_name="unused",
            metrics_ipc_name="unused",
            tokenizer_worker_ipc_name=None,
            decoupled_spec_ipc_config=None,
            instance_id=epoch,
        )
        input_sender = self.context.socket(zmq.PUSH)
        input_sender.bind(ports.scheduler_input_ipc_name)
        self.rejections = self.context.socket(zmq.PULL)
        self.rejections.bind(ports.tokenizer_ipc_name)
        reader = Mock(name="load_snapshot_reader")
        worker_launcher = Mock(name="run_scheduler_process")
        watchdog = Mock(name="soft_watchdog")

        # Keep the real constructor, runtime configuration, ledger, dispatcher,
        # SenderWrapper and ZMQ sockets. Only worker/process startup and the
        # shared-memory reader are replaced. Inproc endpoints cannot reach a
        # service outside this test's context.
        with (
            patch.object(controller_module.zmq, "Context", return_value=self.context),
            patch.object(
                controller_module, "get_zmq_socket", wraps=get_zmq_socket
            ) as socket_factory,
            patch.object(
                controller_module, "create_load_snapshot_reader", return_value=reader
            ) as reader_factory,
            patch.object(
                DataParallelController, "launch_dp_attention_schedulers"
            ) as launch_attention,
            patch.object(
                DataParallelController, "launch_dp_schedulers"
            ) as launch_regular,
            patch.object(controller_module.Watchdog, "create", return_value=watchdog),
            patch.object(controller_module, "start_cpu_monitor_thread") as cpu_monitor,
        ):
            controller = DataParallelController(args, ports, worker_launcher)

        selected_launcher = launch_attention if attention_dp else launch_regular
        other_launcher = launch_regular if attention_dp else launch_attention
        selected_launcher.assert_called_once_with(args, ports)
        other_launcher.assert_not_called()
        reader_factory.assert_called_once_with(ports, caller="DataParallelController")
        cpu_monitor.assert_not_called()
        self.assertIs(controller.load_snapshot_reader, reader)
        self.assertIs(controller.soft_watchdog, watchdog)
        self.assertIs(controller.run_scheduler_process_func, worker_launcher)
        self.assertEqual(controller.load_balance_method, LoadBalanceMethod.TOTAL_TOKENS)
        self.assertEqual(controller.dispatching, controller.total_tokens_scheduler)
        self.assertEqual(controller.dp_budget.num_dp_ranks, 2)
        self.assertEqual(controller.status, [True, True])
        self.assertEqual(controller._active_workers, [0, 1])
        self.assertTrue(callable(controller._request_dispatcher))
        self.assertEqual(
            [call.args[1:] for call in socket_factory.call_args_list],
            [(zmq.PULL, ports.scheduler_input_ipc_name, False)]
            + ([(zmq.PUSH, ports.tokenizer_ipc_name, False)] if admission else []),
        )
        return controller, ports

    def test_admission_off_preserves_both_worker_launch_modes(self):
        for attention_dp in (False, True):
            with self.subTest(attention_dp=attention_dp):
                controller, _ = self.construct(
                    admission=False, attention_dp=attention_dp
                )
                self.assertIsNone(controller.dp_admission)
                self.assertFalse(hasattr(controller, "send_to_tokenizer"))

    def test_admission_on_initializes_ledger_and_real_rejection_sender(self):
        controller, ports = self.construct(admission=True)
        ledger = controller.dp_admission
        self.assertIsInstance(ledger, DPAdmissionController)
        self.assertEqual(ledger.epoch, ports.instance_id)
        self.assertEqual(len(ledger.ranks), 2)
        self.assertEqual(ledger.max_queued_requests, 17)
        self.assertEqual(ledger.queue_timeout, 0.75)
        self.assertEqual(ledger.snapshot_timeout, 13.0)
        self.assertEqual(len(ledger.queue), 0)
        self.assertTrue(
            all(not rank.ready and not rank.pending for rank in ledger.ranks)
        )
        self.assertIsInstance(controller.send_to_tokenizer, SenderWrapper)
        self.assertIs(controller.send_to_tokenizer.socket.context, self.context)

        request = SimpleNamespace(
            rid="constructor-rejection", http_worker_ipc="worker-7"
        )
        controller._reject_admission(
            request, DPAdmissionError(429, "decode queue full")
        )
        self.assertTrue(self.rejections.poll(timeout=1000, flags=zmq.POLLIN))
        rejected = sock_recv(self.rejections, flags=zmq.NOBLOCK)
        self.assertIsInstance(rejected, AbortReq)
        self.assertEqual(rejected.rid, request.rid)
        self.assertEqual(rejected.http_worker_ipc, request.http_worker_ipc)
        self.assertEqual(rejected.finished_reason["status_code"], 429)
        self.assertEqual(rejected.finished_reason["dp_admission_rejected"], 1)
        self.assertEqual(len(ledger.queue), 0)


if __name__ == "__main__":
    unittest.main()
