# SPDX-License-Identifier: Apache-2.0
"""DP admission is opt-in and refuses unaccounted resource configurations."""

import argparse
import unittest

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestDPAdmissionArgs(unittest.TestCase):
    def args(self, **overrides):
        fields = dict(
            model_path="dummy",
            tp_size=2,
            attn_dp_size=2,
            disaggregation_mode="decode",
            load_balance_method="total_tokens",
            disable_radix_cache=True,
            enable_dp_admission=True,
        )
        fields.update(overrides)
        return ServerArgs(**fields)

    def test_default_off_preserves_unrestricted_existing_configs(self):
        args = ServerArgs(model_path="dummy", pp_size=2, dp_size=2)
        args.resolve_once()
        self.assertFalse(resolution_result(args, "enable_dp_admission"))

    def test_supported_configuration_and_legacy_dp_spelling(self):
        for fields in (
            {},
            {"attn_dp_size": 1, "dp_size": 2, "enable_dp_attention": True},
        ):
            with self.subTest(**fields):
                args = self.args(**fields)
                args.resolve_once()
                self.assertTrue(resolution_result(args, "enable_dp_admission"))
                self.assertEqual(resolution_result(args, "attn_dp_size"), 2)

    def test_unsupported_opt_ins_fail_before_model_loading(self):
        variants = (
            {"disaggregation_mode": "prefill"},
            {"attn_dp_size": 1},
            {"nnodes": 2},
            {"pp_size": 2},
            {"dcp_size": 2},
            {"attn_cp_size": 2},
            {"load_balance_method": "round_robin"},
            {"disable_radix_cache": False},
            {"disaggregation_decode_enable_radix_cache": True},
            {"enable_hierarchical_cache": True},
            {"enable_hisparse": True},
            {"disaggregation_decode_host_receive_threshold": 0.8},
            {"enable_pd_role_switch": True},
            {"enable_pdmux": True},
            {"elastic_ep_backend": "mooncake"},
            {"enable_dsa_cache_layer_split": True},
            {"enable_kv_cache_sharding": True},
            {"speculative_algorithm": "DFLASH"},
            {"speculative_eagle_topk": 2},
            {"speculative_adaptive": True},
            {"disaggregation_decode_allocation_policy": "prefill_complete"},
            {"disaggregation_transfer_backend": "nixl"},
            {"is_embedding": True},
            {"enable_lora": True},
            {"smg_grpc_mode": True},
            {"dp_admission_max_queued_requests": -1},
            {"num_reserved_decode_tokens": -1},
            {"dp_admission_queue_timeout": 0},
            {"dp_admission_queue_timeout": float("inf")},
            {"dp_admission_snapshot_timeout": float("nan")},
        )
        for fields in variants:
            with self.subTest(**fields):
                with self.assertRaisesRegex(ValueError, "--enable-dp-admission"):
                    self.args(**fields).resolve_once()

    def test_cli_registers_new_fields(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        parsed = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--enable-dp-admission",
                "--dp-admission-max-queued-requests",
                "4",
                "--dp-admission-queue-timeout",
                "0.25",
                "--dp-admission-snapshot-timeout",
                "15",
            ]
        )
        args = ServerArgs.from_cli_args(parsed)
        self.assertTrue(args.enable_dp_admission)
        self.assertEqual(args.dp_admission_max_queued_requests, 4)
        self.assertEqual(args.dp_admission_queue_timeout, 0.25)
        self.assertEqual(args.dp_admission_snapshot_timeout, 15)


if __name__ == "__main__":
    unittest.main()
