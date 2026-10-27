"""PD failure policy and observability must preserve the release lifecycle."""

import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import zmq

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import KVTransferError
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MODULE = "sglang.srt.disaggregation.mooncake.conn"


class TestMooncakeLegacyObservability(unittest.TestCase):
    def manager(self, metrics=True):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager._mooncake_thread_observability_enabled = True
        manager.server_args = SimpleNamespace(enable_metrics=metrics)
        manager._mooncake_thread_base_labels = {
            "model_name": "glm53",
            "engine_type": "decode",
            "dp_rank": 3,
            "tp_rank": 0,
            "cp_rank": 0,
            "pp_rank": 0,
        }
        manager._mooncake_thread_label_cache = {}
        manager._mooncake_threads = {}
        (
            manager.attn_dp_rank,
            manager.attn_tp_rank,
            manager.attn_cp_rank,
            manager.pp_rank,
        ) = 3, 0, 0, 0
        manager.session_lock = threading.Lock()
        manager.session_failures = defaultdict(int)
        manager.failed_sessions = set()
        manager.session_failure_threshold = 2
        return manager

    def test_only_consecutive_kv_failures_blacklist(self):
        manager = self.manager()
        self.assertFalse(manager._record_session_transfer_failure("peer"))
        self.assertNotIn("peer", manager.failed_sessions)
        manager._reset_session_transfer_failures("peer")
        self.assertFalse(manager._record_session_transfer_failure("peer"))
        self.assertTrue(manager._record_session_transfer_failure("peer"))
        self.assertIn("peer", manager.failed_sessions)
        # Success resets only the sequence; it cannot undo an already-poisoned peer.
        manager._reset_session_transfer_failures("peer")
        self.assertIn("peer", manager.failed_sessions)

    def test_thread_exception_records_failure_and_zero_liveness(self):
        manager = self.manager()
        with (
            patch(MODULE + ".MOONCAKE_THREAD_UP") as up,
            patch(MODULE + ".MOONCAKE_THREAD_EXCEPTIONS") as errors,
            patch(MODULE + ".MOONCAKE_THREAD_LAST_PROGRESS") as progress,
        ):
            with self.assertRaisesRegex(RuntimeError, "receiver crashed"):
                manager._run_monitored_thread(
                    "decode_receiver",
                    "",
                    Mock(side_effect=RuntimeError("receiver crashed")),
                )
            self.assertEqual(
                [x.args[0] for x in up.labels.return_value.set.call_args_list], [1, 0]
            )
            errors.labels.return_value.inc.assert_called_once_with()
            progress.labels.return_value.set.assert_called_once()
            self.assertEqual(up.labels.call_args.kwargs["dp_rank"], 3)

    def test_metrics_disabled_and_other_backends_do_not_publish(self):
        for mode in ("disabled", "other_backend"):
            manager = self.manager(metrics=mode != "disabled")
            manager._mooncake_thread_observability_enabled = mode != "other_backend"
            target = Mock()
            with (
                patch(MODULE + ".MOONCAKE_THREAD_UP") as up,
                patch(MODULE + ".MOONCAKE_THREAD_LAST_PROGRESS") as progress,
            ):
                manager._run_monitored_thread("decode_receiver", "", target)
                target.assert_called_once_with()
                up.labels.assert_not_called()
                progress.labels.assert_not_called()

    def test_started_thread_is_returned_and_joinable_for_role_switch(self):
        manager = self.manager(metrics=False)
        completed = threading.Event()
        thread = manager._start_heartbeat_checker(completed.set)
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(completed.is_set())
        self.assertIs(manager._mooncake_threads[("decode_heartbeat", "")], thread)
        self.assertIn("DP3-TP0-CP0-PP0", thread.name)

    def test_cleanup_during_metadata_parse_does_not_kill_bootstrap(self):
        manager = self.manager(metrics=False)
        manager._stopped = False
        manager.server_socket = object()
        manager.transfer_infos = {}
        manager.req_to_decode_prefix_len = {}
        manager.prefill_bootstrap_timing_lock = threading.Lock()
        manager.prefill_bootstrap_start_times = {17: 1.0}
        manager._worker_threads = []
        manager._handle_abort_notification = lambda _: False
        manager.update_status = Mock()
        message = [b"17", b"host", b"9000", b"peer", b"", b"", b"", b"1"]

        def recv():
            manager._stopped = True
            return message

        manager._make_worker_recv = lambda _: recv
        manager._start_monitored_thread = lambda **kwargs: kwargs["target"]()
        manager.resolve_kv_replica_factor = lambda _: manager.transfer_infos.pop(17)
        transfer = SimpleNamespace(
            dst_kv_indices=np.array([1, 2], dtype=np.int32),
            decode_prefix_len=64,
            is_dummy=False,
        )
        with (
            patch(MODULE + ".TransferInfo.from_zmq", return_value=transfer),
            patch(MODULE + "._log_bootstrap_stage") as logs,
        ):
            manager.start_prefill_thread()
        manager.update_status.assert_called_once_with(17, KVPoll.WaitingForInput)
        self.assertEqual(manager.req_to_decode_prefix_len[17], 64)
        self.assertNotIn(17, manager.prefill_bootstrap_start_times)
        final = logs.call_args
        self.assertEqual(final.args[:3], ("Prefill", "metadata_wait", "end"))
        self.assertEqual(final.kwargs["total_kv_indices_count"], 2)

    def receiver(self):
        receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
        receiver.bootstrap_room = 17
        receiver.session_id = "peer"
        receiver.bootstrap_started_at = 0.0
        receiver.bootstrap_infos = [
            {"is_dummy": False, "rank_ip": "host", "rank_port": 9000}
        ]
        receiver.required_dst_info_num = 1
        receiver.init_time = None
        receiver.conclude_state = None
        receiver.kv_mgr = SimpleNamespace(
            enable_staging=False,
            local_ip="decode",
            rank_port=8000,
            record_failure=Mock(),
            update_status=Mock(),
        )
        receiver.invalidate_cached_bootstrap_infos = Mock()
        socket = Mock()
        receiver._connect_to_bootstrap_server = lambda _: (socket, threading.Lock())
        return receiver, socket

    def test_metadata_logging_keeps_release_payload_and_single_send(self):
        receiver, socket = self.receiver()
        kv_indices = np.array([3, 7], dtype=np.int32)
        device_indices = np.array([12, 13], dtype=np.int32)
        with patch(MODULE + "._log_bootstrap_stage") as logs:
            receiver.send_metadata(kv_indices, 5, [[6]], 64, device_indices)
        socket.send_multipart.assert_called_once()
        wire = socket.send_multipart.call_args.args[0]
        self.assertEqual(len(wire), 10)
        self.assertEqual(wire[4], kv_indices.tobytes())
        self.assertEqual(wire[8], b"64")
        self.assertEqual(wire[9], device_indices.tobytes())
        self.assertIsNotNone(receiver.init_time)
        self.assertEqual(logs.call_args.kwargs["result"], "metadata_sent")

    def test_metadata_failure_keeps_failure_state_and_ends_stage(self):
        receiver, socket = self.receiver()
        socket.send_multipart.side_effect = zmq.ZMQError("disconnected")
        with patch(MODULE + "._log_bootstrap_stage") as logs:
            receiver.send_metadata(np.array([3], dtype=np.int32), 5)
        receiver.invalidate_cached_bootstrap_infos.assert_called_once()
        receiver.kv_mgr.update_status.assert_called_once_with(17, KVPoll.Failed)
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)
        self.assertIsNone(receiver.init_time)
        self.assertEqual(logs.call_args.args[1:3], ("metadata_send", "end"))
        self.assertEqual(logs.call_args.kwargs["result"], "exception")

    def test_receiver_failure_does_not_use_sender_only_wait_state(self):
        receiver, _ = self.receiver()
        receiver.clear = Mock()
        receiver.kv_mgr.failure_lock = threading.Lock()
        receiver.kv_mgr.failure_records = {17: "transport failed"}
        with self.assertRaises(KVTransferError):
            receiver.failure_exception()
        receiver.clear.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
