# SPDX-License-Identifier: Apache-2.0
"""Deterministic, explicitly calibrated compute budget for one prefill batch.

All lengths are logical tokens for the complete attention TP/CP group. There is
no rank-local clock or online fitting: ranks with the same request state and
calibration select the same chunks. This is an estimated compute budget, not a
wall-clock deadline or a replacement for the token and memory admission limits.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from typing import Optional

_MODEL_FLAG = "--prefill-time-model"


def _finite_number(value) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class PrefillTimeModel:
    fixed_ms: float = 0.0
    linear_ms_per_token: float = 0.0
    quadratic_ms_per_token2: float = 0.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not _finite_number(value) or value < 0:
                raise ValueError(
                    f"{_MODEL_FLAG}: {field.name} must be finite and nonnegative."
                )
            # JSON may encode a large finite coefficient as an integer. Keep
            # arithmetic floating-point so overflow is conservative infinity,
            # rather than an int-to-float OverflowError in the scheduler.
            object.__setattr__(self, field.name, float(value))
        if self.linear_ms_per_token == self.quadratic_ms_per_token2 == 0:
            raise ValueError(
                f"{_MODEL_FLAG} requires a positive linear_ms_per_token "
                "or quadratic_ms_per_token2."
            )

    @classmethod
    def from_json(cls, value: Optional[str]) -> Optional[PrefillTimeModel]:
        if value is None:
            return None
        try:
            config = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{_MODEL_FLAG} must be a JSON object: {exc}") from exc
        if not isinstance(config, dict):
            raise ValueError(f"{_MODEL_FLAG} must be a JSON object.")
        unknown = config.keys() - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"{_MODEL_FLAG}: unknown fields {sorted(unknown)}.")
        return cls(**config)

    def token_cost_ms(self, prefix_len: int, extend_len: int) -> float:
        if prefix_len < 0 or extend_len < 0:
            raise ValueError("Prefill lengths must be nonnegative.")
        if extend_len == 0:
            return 0.0
        # This form avoids subtracting two large, almost-equal squared lengths.
        return extend_len * (
            self.linear_ms_per_token
            + self.quadratic_ms_per_token2 * (2 * prefix_len + extend_len)
        )


class PrefillTimeBudget:
    def __init__(self, model: PrefillTimeModel, budget_ms: float):
        if not _finite_number(budget_ms) or budget_ms <= 0:
            raise ValueError("--prefill-time-budget-ms must be finite and positive.")
        self.model = model
        self.budget_ms = float(budget_ms)
        self.used_ms = 0.0
        self.admitted_requests = 0

    @property
    def remaining_ms(self) -> float:
        return self.budget_ms - self.used_ms

    @property
    def exhausted(self) -> bool:
        # The first request may need a minimum progress chunk even when the
        # fixed batch overhead alone exceeds the requested soft budget.
        return self.admitted_requests > 0 and self.remaining_ms <= 0

    def fit_tokens(self, prefix_len: int, token_limit: int) -> int:
        """Largest integer work size that fits; selection never spends budget."""
        if prefix_len < 0 or token_limit < 0:
            raise ValueError("Prefill lengths must be nonnegative.")
        available = self.remaining_ms
        if self.admitted_requests == 0:
            available -= self.model.fixed_ms
        if available <= 0:
            return 0

        # A bounded integer search is monotone for nonnegative coefficients and
        # remains conservative at floating-point boundaries and on overflow.
        low, high = 0, token_limit
        while low < high:
            middle = (low + high + 1) // 2
            if self.model.token_cost_ms(prefix_len, middle) <= available:
                low = middle
            else:
                high = middle - 1
        return low

    def select_tokens(
        self,
        *,
        prefix_len: int,
        remaining_tokens: int,
        token_limit: int,
        alignment: int,
        allow_minimum: bool,
        extend_alignment: int = 1,
    ) -> int:
        """Cap a request while respecting both backend alignment constraints.

        ``alignment`` constrains the absolute chunk end (KV pages), whereas
        ``extend_alignment`` constrains the query length (attention split tiles
        or Mamba chunks). A finishing tail need not align. Minimum progress may
        exceed only the time estimate; it never exceeds ``token_limit``.
        """
        if (
            prefix_len < 0
            or remaining_tokens < 0
            or token_limit < 0
            or alignment < 1
            or extend_alignment < 1
        ):
            raise ValueError("Invalid prefill token limit or alignment.")
        hard_limit = min(remaining_tokens, token_limit)
        fit = self.fit_tokens(prefix_len, hard_limit)
        if fit >= remaining_tokens:
            return remaining_tokens

        # q = 0 (mod extend_alignment), prefix + q = 0 (mod alignment).
        # Combining these as an absolute-end LCM loses query-length alignment
        # for a nonzero prefix (e.g. prefix=64, page=64, split tile=4096).
        common = math.gcd(alignment, extend_alignment)
        minimum = remaining_tokens
        if prefix_len % common == 0:
            period = math.lcm(alignment, extend_alignment)
            modulus = alignment // common
            multiplier = (
                ((-prefix_len // common) * pow(extend_alignment // common, -1, modulus))
                % modulus
                if modulus > 1
                else 0
            )
            first = extend_alignment * multiplier or period
            if fit >= first:
                return first + (fit - first) // period * period
            minimum = min(remaining_tokens, first)

        if allow_minimum and self.admitted_requests == 0:
            if minimum <= hard_limit:
                return minimum
        return 0

    def charge(self, prefix_len: int, extend_len: int) -> None:
        """Spend once, only for the final admitted request shape."""
        cost = self.model.token_cost_ms(prefix_len, extend_len)
        if self.admitted_requests == 0:
            cost += self.model.fixed_ms
        self.used_ms += cost
        self.admitted_requests += 1
