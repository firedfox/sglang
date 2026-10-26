# SPDX-License-Identifier: Apache-2.0
"""CPU-only cost estimates for the optional L3 prefetch stop policy.

Rates describe one request's logical tokens for the complete cache group. They
must be calibrated for the model, context lengths and TP/CP topology in use;
there are deliberately no default throughput estimates. Observed, rank-agreed
transfer progress supersedes a configured read rate once a sample is available.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from typing import Optional

_FLAG = "--hicache-storage-prefetch-cost-model"


@dataclass(frozen=True)
class PrefetchCostEstimate:
    remaining_read_seconds: float
    recompute_seconds: float
    should_cancel: bool


@dataclass(frozen=True)
class PrefetchCostModel:
    recompute_tokens_per_second: float
    read_tokens_per_second: Optional[float] = None
    min_observation_seconds: float = 0.05
    cancel_margin: float = 1.2
    recompute_fixed_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in ("recompute_tokens_per_second", "read_tokens_per_second"):
            value = getattr(self, name)
            if name == "read_tokens_per_second" and value is None:
                continue
            if not self._finite_number(value) or value <= 0:
                raise ValueError(f"{_FLAG}: {name} must be finite and positive.")
        for name in ("min_observation_seconds", "recompute_fixed_seconds"):
            value = getattr(self, name)
            if not self._finite_number(value) or value < 0:
                raise ValueError(f"{_FLAG}: {name} must be finite and nonnegative.")
        if not self._finite_number(self.cancel_margin) or self.cancel_margin < 1:
            raise ValueError(f"{_FLAG}: cancel_margin must be finite and at least 1.")

    @staticmethod
    def _finite_number(value) -> bool:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        try:
            return math.isfinite(value)
        except OverflowError:
            return False

    @classmethod
    def from_json(cls, value: Optional[str]) -> Optional[PrefetchCostModel]:
        if value is None:
            return None
        try:
            config = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{_FLAG} must be a JSON object: {exc}") from exc
        if not isinstance(config, dict):
            raise ValueError(f"{_FLAG} must be a JSON object.")
        unknown = config.keys() - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"{_FLAG}: unknown fields {sorted(unknown)}.")
        if "recompute_tokens_per_second" not in config:
            raise ValueError(f"{_FLAG} requires recompute_tokens_per_second.")
        return cls(**config)

    def estimate(
        self,
        *,
        remaining_tokens: int,
        recompute_tokens: int,
        observed_tokens: int,
        observation_seconds: float,
    ) -> Optional[PrefetchCostEstimate]:
        """Compare future read time with recomputing the unusable prefix.

        A zero-progress sample is not an infinite-time estimate: asynchronous
        storage ACKs may be coarse. With neither positive progress nor a
        calibrated read rate the caller must fall back to its timeout policy.
        Already spent waiting time is sunk cost, so it is not added to either
        alternative. The observation window resets when a transfer is reissued
        or its logical span changes; the caller's request deadline does not.
        """
        if (
            remaining_tokens <= 0
            or recompute_tokens <= 0
            or observed_tokens < 0
            or not self._finite_number(observation_seconds)
            or observation_seconds < self.min_observation_seconds
        ):
            return None

        read_rate = self.read_tokens_per_second
        if observed_tokens > 0 and observation_seconds > 0:
            read_rate = observed_tokens / observation_seconds
        if read_rate is None or not self._finite_number(read_rate) or read_rate <= 0:
            return None

        remaining_read = remaining_tokens / read_rate
        recompute = (
            self.recompute_fixed_seconds
            + recompute_tokens / self.recompute_tokens_per_second
        )
        threshold = self.cancel_margin * recompute
        if not all(
            math.isfinite(value) for value in (remaining_read, recompute, threshold)
        ):
            return None
        return PrefetchCostEstimate(
            remaining_read_seconds=remaining_read,
            recompute_seconds=recompute,
            should_cancel=remaining_read > threshold,
        )

    def describe_estimate(
        self,
        estimate: Optional[PrefetchCostEstimate],
        *,
        remaining_tokens: int,
        recompute_tokens: int,
        observed_tokens: int,
        observation_seconds: float,
    ) -> dict[str, object]:
        """Explain an existing estimate without evaluating the policy again.

        The caller supplies the same sample used for ``estimate``. Diagnostics
        must not read another clock or turn an unavailable estimate into a
        keep decision. No request identifiers become metric labels.
        """
        if estimate is not None:
            reason = (
                "read_exceeds_recompute_margin"
                if estimate.should_cancel
                else "read_within_recompute_margin"
            )
        elif remaining_tokens <= 0 or recompute_tokens <= 0:
            reason = "no_remaining_work"
        elif observed_tokens < 0:
            reason = "invalid_progress"
        elif not self._finite_number(observation_seconds):
            reason = "invalid_observation"
        elif observation_seconds < self.min_observation_seconds:
            reason = "observation_too_short"
        elif self.read_tokens_per_second is None and not (
            observed_tokens > 0 and observation_seconds > 0
        ):
            reason = "no_read_rate"
        else:
            reason = "nonfinite_estimate"

        rate_source = "unavailable"
        read_rate = None
        if estimate is not None:
            if observed_tokens > 0 and observation_seconds > 0:
                rate_source = "observed"
                read_rate = observed_tokens / observation_seconds
            else:
                rate_source = "configured"
                read_rate = self.read_tokens_per_second
        return {
            "decision": (
                "timeout_fallback"
                if estimate is None
                else "cancel"
                if estimate.should_cancel
                else "keep"
            ),
            "reason": reason,
            "remaining_tokens": remaining_tokens,
            "recompute_tokens": recompute_tokens,
            "observed_tokens": observed_tokens,
            "observation_seconds": (
                observation_seconds
                if self._finite_number(observation_seconds)
                else None
            ),
            "min_observation_seconds": self.min_observation_seconds,
            "read_rate_source": rate_source,
            "read_tokens_per_second": read_rate,
            "configured_read_tokens_per_second": self.read_tokens_per_second,
            "recompute_tokens_per_second": self.recompute_tokens_per_second,
            "cancel_margin": self.cancel_margin,
            "recompute_fixed_seconds": self.recompute_fixed_seconds,
            "remaining_read_seconds": (
                estimate.remaining_read_seconds if estimate is not None else None
            ),
            "recompute_seconds": (
                estimate.recompute_seconds if estimate is not None else None
            ),
            "cancel_threshold_seconds": (
                self.cancel_margin * estimate.recompute_seconds
                if estimate is not None
                else None
            ),
        }
