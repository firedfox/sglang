# SPDX-License-Identifier: Apache-2.0
"""L3 wait-vs-recompute decisions without a cache, distributed runtime or GPU."""

import json
import unittest

from sglang.srt.mem_cache.prefetch_cost import PrefetchCostModel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPrefetchCostModel(unittest.TestCase):
    def estimate(self, model, **overrides):
        state = dict(
            remaining_tokens=1000,
            recompute_tokens=1000,
            observed_tokens=0,
            observation_seconds=1.0,
        )
        state.update(overrides)
        return model.estimate(**state)

    def test_no_calibration_is_an_explicit_timeout_fallback(self):
        self.assertIsNone(PrefetchCostModel.from_json(None))
        model = PrefetchCostModel(recompute_tokens_per_second=1000)
        self.assertIsNone(self.estimate(model))

    def test_calibrated_slow_read_cancels_but_fast_read_keeps_waiting(self):
        for read_rate, expected in ((100, True), (10000, False)):
            with self.subTest(read_rate=read_rate):
                model = PrefetchCostModel(
                    recompute_tokens_per_second=1000,
                    read_tokens_per_second=read_rate,
                )
                estimate = self.estimate(model)
                self.assertEqual(estimate.should_cancel, expected)
                self.assertEqual(estimate.remaining_read_seconds, 1000 / read_rate)
                self.assertEqual(estimate.recompute_seconds, 1)

    def test_observed_progress_supersedes_stale_read_calibration(self):
        for calibrated, observed, expected in (
            (10000, 100, True),
            (100, 10000, False),
            (None, 100, True),
            (None, 10000, False),
        ):
            with self.subTest(calibrated=calibrated, observed=observed):
                model = PrefetchCostModel(
                    recompute_tokens_per_second=1000,
                    read_tokens_per_second=calibrated,
                )
                estimate = self.estimate(model, observed_tokens=observed)
                self.assertEqual(estimate.should_cancel, expected)

    def test_stalled_progress_does_not_create_infinite_read_estimate(self):
        model = PrefetchCostModel(recompute_tokens_per_second=1000)
        self.assertIsNone(self.estimate(model, observation_seconds=1000))

    def test_minimum_observation_applies_to_calibrated_reads_too(self):
        model = PrefetchCostModel(
            recompute_tokens_per_second=1000,
            read_tokens_per_second=1,
            min_observation_seconds=0.1,
        )
        self.assertIsNone(self.estimate(model, observation_seconds=0.099))
        self.assertTrue(self.estimate(model, observation_seconds=0.1).should_cancel)

    def test_cancel_margin_avoids_small_differences(self):
        model = PrefetchCostModel(
            recompute_tokens_per_second=1000,
            read_tokens_per_second=800,
            cancel_margin=1.25,
        )
        self.assertFalse(self.estimate(model).should_cancel)
        self.assertTrue(self.estimate(model, observed_tokens=799).should_cancel)

    def test_nearly_finished_read_can_beat_recompute_setup_cost(self):
        model = PrefetchCostModel(
            recompute_tokens_per_second=1000,
            read_tokens_per_second=100,
            recompute_fixed_seconds=0.02,
        )
        self.assertTrue(self.estimate(model).should_cancel)
        self.assertFalse(
            self.estimate(model, remaining_tokens=1, recompute_tokens=1).should_cancel
        )

    def test_hybrid_discard_compares_whole_unusable_prefix(self):
        model = PrefetchCostModel(
            recompute_tokens_per_second=1000, read_tokens_per_second=200
        )
        self.assertTrue(
            self.estimate(
                model, remaining_tokens=100, recompute_tokens=100
            ).should_cancel
        )
        self.assertFalse(
            self.estimate(
                model, remaining_tokens=100, recompute_tokens=1000
            ).should_cancel
        )

    def test_invalid_runtime_samples_fall_back_without_cancelling(self):
        model = PrefetchCostModel(
            recompute_tokens_per_second=1000, read_tokens_per_second=100
        )
        for sample in (
            {"remaining_tokens": 0},
            {"recompute_tokens": 0},
            {"observed_tokens": -1},
            {"observation_seconds": -1},
            {"observation_seconds": float("nan")},
            {"observation_seconds": float("inf")},
        ):
            with self.subTest(sample=sample):
                self.assertIsNone(self.estimate(model, **sample))

    def test_arithmetic_overflow_is_not_a_valid_cost_estimate(self):
        for config in (
            {"recompute_tokens_per_second": 1e-320, "read_tokens_per_second": 1},
            {"recompute_tokens_per_second": 1, "read_tokens_per_second": 1e-320},
            {
                "recompute_tokens_per_second": 1,
                "read_tokens_per_second": 1,
                "cancel_margin": 1e308,
            },
        ):
            with self.subTest(config=config):
                self.assertIsNone(self.estimate(PrefetchCostModel(**config)))

    def test_json_schema_rejects_typos_and_non_objects(self):
        for raw in ("{", "[]", "null", '"model"', "{}", '{"read_tps": 1}'):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, "prefetch-cost-model"):
                    PrefetchCostModel.from_json(raw)

    def test_nonfinite_negative_and_boolean_calibration_are_rejected(self):
        valid = {"recompute_tokens_per_second": 1000}
        invalid = (
            None,
            False,
            True,
            0,
            -1,
            "1000",
            float("nan"),
            float("inf"),
            10**400,
        )
        for field in ("recompute_tokens_per_second", "read_tokens_per_second"):
            for value in invalid:
                if field == "read_tokens_per_second" and value is None:
                    continue
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field):
                        PrefetchCostModel.from_json(json.dumps(valid | {field: value}))
        for field in ("min_observation_seconds", "recompute_fixed_seconds"):
            for value in (False, True, -1, "1", None, float("nan"), float("inf")):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field):
                        PrefetchCostModel.from_json(json.dumps(valid | {field: value}))
        for margin in (0.9, False, "1", None, float("nan"), float("inf")):
            with self.subTest(margin=margin):
                with self.assertRaisesRegex(ValueError, "cancel_margin"):
                    PrefetchCostModel.from_json(
                        json.dumps(valid | {"cancel_margin": margin})
                    )

    def test_zero_setup_and_observation_are_allowed(self):
        model = PrefetchCostModel.from_json(
            '{"recompute_tokens_per_second":1000,"read_tokens_per_second":100,'
            '"min_observation_seconds":0,"recompute_fixed_seconds":0,"cancel_margin":1}'
        )
        self.assertTrue(self.estimate(model, observation_seconds=0).should_cancel)


if __name__ == "__main__":
    unittest.main()
