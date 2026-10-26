# SPDX-License-Identifier: Apache-2.0
"""Deterministic prefill cost and aligned progress without GPU timing."""

import json
import unittest
from unittest.mock import patch

from sglang.srt.managers.scheduler_components.prefill_time_budget import (
    PrefillTimeBudget,
    PrefillTimeModel,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPrefillTimeBudget(unittest.TestCase):
    def test_model_uses_prefix_attention_work_without_large_square_subtraction(self):
        model = PrefillTimeModel(linear_ms_per_token=2, quadratic_ms_per_token2=0.5)
        self.assertEqual(model.token_cost_ms(10, 4), 56)
        self.assertEqual(model.token_cost_ms(10**12, 1), 10**12 + 2.5)

    def test_fixed_cost_is_charged_once_per_batch(self):
        budget = PrefillTimeBudget(
            PrefillTimeModel(fixed_ms=3, linear_ms_per_token=2), 13
        )
        self.assertEqual(budget.fit_tokens(0, 100), 5)
        budget.charge(0, 2)
        self.assertEqual(budget.used_ms, 7)
        self.assertEqual(budget.fit_tokens(100, 100), 3)
        budget.charge(100, 3)
        self.assertEqual(budget.used_ms, 13)
        self.assertTrue(budget.exhausted)

    def test_repeated_selection_does_not_spend_budget(self):
        budget = PrefillTimeBudget(PrefillTimeModel(linear_ms_per_token=1), 16)
        for _ in range(4):
            self.assertEqual(budget.fit_tokens(32, 100), 16)
        self.assertEqual((budget.used_ms, budget.admitted_requests), (0, 0))

    def test_longer_prefix_never_increases_fitting_chunk(self):
        budget = PrefillTimeBudget(PrefillTimeModel(quadratic_ms_per_token2=0.01), 100)
        chunks = [
            budget.fit_tokens(prefix, 1 << 20) for prefix in (0, 32, 1024, 1 << 20)
        ]
        self.assertEqual(chunks, sorted(chunks, reverse=True))
        for prefix, chunk in zip((0, 32, 1024, 1 << 20), chunks):
            self.assertLessEqual(budget.model.token_cost_ms(prefix, chunk), 100)
            self.assertGreater(budget.model.token_cost_ms(prefix, chunk + 1), 100)

    def test_absolute_end_alignment_and_finishing_tail(self):
        budget = PrefillTimeBudget(PrefillTimeModel(linear_ms_per_token=1), 70)
        self.assertEqual(
            budget.select_tokens(
                prefix_len=3,
                remaining_tokens=1000,
                token_limit=100,
                alignment=8,
                allow_minimum=False,
            ),
            69,
        )
        self.assertEqual(
            budget.select_tokens(
                prefix_len=3,
                remaining_tokens=2,
                token_limit=100,
                alignment=8,
                allow_minimum=False,
            ),
            2,
        )

    def test_minimum_progress_only_escapes_time_and_only_once(self):
        budget = PrefillTimeBudget(
            PrefillTimeModel(fixed_ms=2, linear_ms_per_token=1), 0.1
        )
        self.assertFalse(budget.exhausted)
        args = dict(
            prefix_len=3, remaining_tokens=1000, alignment=8, allow_minimum=True
        )
        self.assertEqual(budget.select_tokens(token_limit=4, **args), 0)
        self.assertEqual(budget.select_tokens(token_limit=100, **args), 5)
        budget.charge(3, 5)
        self.assertTrue(budget.exhausted)
        self.assertEqual(budget.select_tokens(token_limit=100, **args), 0)

    def test_short_final_tail_can_make_progress_below_alignment(self):
        budget = PrefillTimeBudget(PrefillTimeModel(linear_ms_per_token=1), 0.1)
        self.assertEqual(
            budget.select_tokens(
                prefix_len=63,
                remaining_tokens=2,
                token_limit=2,
                alignment=128,
                allow_minimum=True,
            ),
            2,
        )

    def test_relative_query_alignment_is_not_absolute_end_alignment(self):
        budget = PrefillTimeBudget(PrefillTimeModel(linear_ms_per_token=1), 8192)
        self.assertEqual(
            budget.select_tokens(
                prefix_len=64,
                remaining_tokens=20000,
                token_limit=16384,
                alignment=64,
                extend_alignment=4096,
                allow_minimum=False,
            ),
            8192,
        )

    def test_compatible_nonzero_residue_satisfies_both_alignments(self):
        budget = PrefillTimeBudget(PrefillTimeModel(linear_ms_per_token=1), 65)
        # q is divisible by 6 and prefix + q is divisible by 8: 18, 42, 66, ...
        self.assertEqual(
            budget.select_tokens(
                prefix_len=6,
                remaining_tokens=100,
                token_limit=100,
                alignment=8,
                extend_alignment=6,
                allow_minimum=False,
            ),
            42,
        )

    def test_incompatible_alignments_never_create_an_illegal_cut(self):
        budget = PrefillTimeBudget(PrefillTimeModel(linear_ms_per_token=1), 1)
        # Streaming sessions that could create these prefixes are rejected at
        # configuration validation. The pure selector still must not make an
        # illegal intermediate cut if independently called with such a prefix.
        args = dict(
            prefix_len=3, alignment=64, extend_alignment=4096, allow_minimum=True
        )
        self.assertEqual(
            budget.select_tokens(remaining_tokens=20000, token_limit=8192, **args), 0
        )
        self.assertEqual(
            budget.select_tokens(remaining_tokens=2, token_limit=8192, **args), 2
        )

    def test_cp_ranks_use_one_group_budget_without_clocks_or_division(self):
        model = PrefillTimeModel(linear_ms_per_token=0.25)
        with patch("time.perf_counter", side_effect=AssertionError("no wall clock")):
            sizes = [
                PrefillTimeBudget(model, 16).fit_tokens(1 << 20, 131072)
                for _ in range(8)
            ]
        self.assertEqual(sizes, [64] * 8)

    def test_overflow_is_conservatively_nonfitting(self):
        for model in (
            PrefillTimeModel(quadratic_ms_per_token2=1e308),
            PrefillTimeModel(quadratic_ms_per_token2=10**308),
            PrefillTimeModel.from_json(
                json.dumps({"quadratic_ms_per_token2": 10**308})
            ),
        ):
            with self.subTest(model=model):
                budget = PrefillTimeBudget(model, 1)
                self.assertEqual(budget.fit_tokens(1 << 20, 131072), 0)
                self.assertEqual(model.token_cost_ms(1 << 20, 0), 0)

    def test_json_supports_linear_and_quadratic_calibrations(self):
        self.assertIsNone(PrefillTimeModel.from_json(None))
        self.assertEqual(
            PrefillTimeModel.from_json('{"linear_ms_per_token":1}'),
            PrefillTimeModel(linear_ms_per_token=1),
        )
        self.assertEqual(
            PrefillTimeModel.from_json('{"quadratic_ms_per_token2":1}'),
            PrefillTimeModel(quadratic_ms_per_token2=1),
        )

    def test_invalid_models_are_rejected(self):
        for raw in (
            "{",
            "[]",
            "null",
            "{}",
            '{"fixed_ms":1}',
            '{"linear_ms_per_token":1,"unknown":0}',
        ):
            with (
                self.subTest(raw=raw),
                self.assertRaisesRegex(ValueError, "prefill-time-model"),
            ):
                PrefillTimeModel.from_json(raw)
        for field in ("fixed_ms", "linear_ms_per_token", "quadratic_ms_per_token2"):
            for value in (-1, None, True, "1", float("nan"), float("inf"), 10**400):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaisesRegex(ValueError, field),
                ):
                    PrefillTimeModel.from_json(
                        json.dumps({"linear_ms_per_token": 1, field: value})
                    )

    def test_invalid_budget_is_rejected(self):
        for value in (0, -1, True, None, "1", float("nan"), float("inf")):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "prefill-time-budget-ms"),
            ):
                PrefillTimeBudget(PrefillTimeModel(linear_ms_per_token=1), value)


if __name__ == "__main__":
    unittest.main()
