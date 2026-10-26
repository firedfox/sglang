# SPDX-License-Identifier: Apache-2.0
"""Prefill time budgeting is explicitly calibrated and rejected where unmodeled."""

import argparse
import unittest

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_MODEL = '{"fixed_ms":1,"linear_ms_per_token":0.01,"quadratic_ms_per_token2":1e-8}'


class TestPrefillTimeBudgetArgs(unittest.TestCase):
    def args(self, **overrides):
        fields = dict(
            model_path="dummy",
            prefill_time_budget_ms=100,
            prefill_time_model=_MODEL,
            chunked_prefill_size=8192,
        )
        fields.update(overrides)
        return ServerArgs(**fields)

    def test_defaults_preserve_existing_incompatible_modes(self):
        args = ServerArgs(model_path="dummy", pp_size=2, enable_mixed_chunk=True)
        args.resolve_once()
        self.assertIsNone(resolution_result(args, "prefill_time_budget_ms"))
        self.assertIsNone(resolution_result(args, "prefill_time_model"))

    def test_pair_and_invalid_values_fail_before_model_loading(self):
        for fields in (
            {"prefill_time_budget_ms": None},
            {"prefill_time_model": None},
            {"prefill_time_budget_ms": 0},
            {"prefill_time_budget_ms": float("nan")},
            {"prefill_time_budget_ms": float("inf")},
            {"prefill_time_model": "{}"},
            {"prefill_time_model": '{"linear_ms_per_token":-1}'},
        ):
            with (
                self.subTest(fields=fields),
                self.assertRaisesRegex(ValueError, "prefill-time"),
            ):
                self.args(**fields).resolve_once()

    def test_unsupported_paths_are_rejected(self):
        for fields in (
            {"pp_size": 2},
            {"enable_dynamic_chunking": True},
            {"enable_mixed_chunk": True},
            {"dllm_algorithm": "LowConfidence"},
            {"disaggregation_mode": "decode"},
            {"is_embedding": True},
            {"enable_encoder_swa_bounded_replay": True},
            {"enable_streaming_session": True},
            {"chunked_prefill_size": -1},
            {"chunked_prefill_size": 0},
            {"max_prefill_tokens": 0},
        ):
            with (
                self.subTest(fields=fields),
                self.assertRaisesRegex(ValueError, "prefill-time-budget-ms"),
            ):
                self.args(**fields).resolve_once()

    def test_group_calibration_is_not_rescaled_by_cp(self):
        args = self.args(tp_size=8, attn_cp_size=8, disaggregation_mode="prefill")
        args.resolve_once()
        self.assertEqual(resolution_result(args, "prefill_time_budget_ms"), 100)
        self.assertEqual(resolution_result(args, "prefill_time_model"), _MODEL)

    def test_cli_registers_both_fields(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = ServerArgs.from_cli_args(
            parser.parse_args(
                [
                    "--model-path",
                    "dummy",
                    "--prefill-time-budget-ms",
                    "25.5",
                    "--prefill-time-model",
                    _MODEL,
                    "--chunked-prefill-size",
                    "8192",
                ]
            )
        )
        args.resolve_once()
        self.assertEqual(args.prefill_time_budget_ms, 25.5)
        self.assertEqual(args.prefill_time_model, _MODEL)


if __name__ == "__main__":
    unittest.main()
