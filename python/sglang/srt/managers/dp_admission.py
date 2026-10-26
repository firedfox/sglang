# SPDX-License-Identifier: Apache-2.0
"""Decode admission accounting shared by the DP controller and schedulers.

An ordinary load timestamp is not an acknowledgement: a newer snapshot can
precede requests already sent by the controller. Each rank therefore publishes
the last dispatch sequence it has processed together with its *live* commitments.
The controller adds only reservations beyond that watermark. Cancellation never
refunds a dispatched reservation; the scheduler owns it until the request is
terminal and its KV allocation has actually been released.

This module deliberately has no GPU, IPC, or server-configuration dependencies.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


class DPAdmissionError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class DPAdmissionRequest:
    prompt_tokens: int
    max_new_tokens: int | None
    routed_dp_rank: int | None = None


@dataclass(frozen=True)
class DPAdmissionReservation:
    rank: int
    sequence: int
    tokens: int


@dataclass
class _RankBudget:
    timestamp: float = 0.0
    received_at: float = float("-inf")
    ready: bool = False
    sequence: int = 0
    acknowledged: int = 0
    token_capacity: int = 0
    request_capacity: int = 0
    used_tokens: int = 0
    used_requests: int = 0
    page_size: int = 0
    decode_reserve: int = 0
    prealloc_reserve: int = 0
    max_sequence_length: int = 0
    max_new_tokens_limit: int = 0
    pending: dict[int, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.used_tokens + sum(self.pending.values())

    @property
    def total_requests(self) -> int:
        return self.used_requests + len(self.pending)

    def charge(self, request: DPAdmissionRequest) -> int:
        # Match the scheduler's context bound, without modifying sampling params.
        remaining = self.max_sequence_length - request.prompt_tokens - 1
        if remaining < 0:
            raise DPAdmissionError(
                400, "Request prompt exceeds the rank context limit."
            )
        output = (
            remaining
            if request.max_new_tokens is None
            else min(remaining, request.max_new_tokens)
        )
        if self.max_new_tokens_limit > 0:
            output = min(output, self.max_new_tokens_limit)
        prompt_allocation = (
            (request.prompt_tokens + self.page_size - 1)
            // self.page_size
            * self.page_size
        )
        tokens = max(
            request.prompt_tokens + output + self.decode_reserve,
            prompt_allocation + self.prealloc_reserve,
        )
        return (tokens + self.page_size - 1) // self.page_size * self.page_size


@dataclass(frozen=True)
class QueuedDPRequest:
    request: Any
    cost: DPAdmissionRequest
    deadline: float


class DPAdmissionController:
    """Per-rank, acknowledgement-based reservations and a bounded FIFO queue."""

    def __init__(
        self,
        num_ranks: int,
        epoch: str,
        *,
        max_queued_requests: int = 0,
        queue_timeout: float = 1.0,
        snapshot_timeout: float = 30.0,
    ):
        if not epoch:
            raise ValueError("DP admission requires a nonempty server instance ID.")
        self.epoch = epoch
        self.ranks = [_RankBudget() for _ in range(num_ranks)]
        self.max_queued_requests = max_queued_requests
        self.queue_timeout = queue_timeout
        self.snapshot_timeout = snapshot_timeout
        self.queue: deque[QueuedDPRequest] = deque()

    def update(self, loads, now: float) -> None:
        for load in loads:
            if not 0 <= load.dp_rank < len(self.ranks):
                continue
            rank = self.ranks[load.dp_rank]
            if load.timestamp <= rank.timestamp:
                continue
            rank.timestamp = load.timestamp
            rank.received_at = now
            ack = load.dp_admission_sequence
            rank.ready = (
                load.dp_admission_epoch == self.epoch
                and rank.acknowledged <= ack <= rank.sequence
                and load.max_total_num_tokens > 0
                and load.max_running_requests > 0
                and load.dp_admission_page_size > 0
                and load.dp_admission_max_seq_len > 0
                and load.dp_admission_decode_reserve >= 0
                and load.dp_admission_prealloc_reserve >= 0
                and load.dp_admission_reserved_tokens >= 0
                and load.dp_admission_reserved_requests >= 0
            )
            if not rank.ready:
                # Do not discard reservations when the source cannot ACK them.
                continue
            rank.acknowledged = ack
            rank.pending = {
                seq: tokens for seq, tokens in rank.pending.items() if seq > ack
            }
            rank.token_capacity = load.max_total_num_tokens
            rank.request_capacity = load.max_running_requests
            # Full commitments already include active and preallocated KV once.
            # The observed floors also retain allocator frees deferred to a batch
            # boundary and fail conservatively for any untracked occupancy.
            rank.used_tokens = max(
                load.dp_admission_reserved_tokens, load.num_total_tokens
            )
            rank.used_requests = max(
                load.dp_admission_reserved_requests,
                load.num_running_reqs + load.num_waiting_reqs,
            )
            rank.page_size = load.dp_admission_page_size
            rank.decode_reserve = load.dp_admission_decode_reserve
            rank.prealloc_reserve = load.dp_admission_prealloc_reserve
            rank.max_sequence_length = load.dp_admission_max_seq_len
            rank.max_new_tokens_limit = load.dp_admission_max_new_tokens_limit

    def choose(
        self,
        request: DPAdmissionRequest,
        active_ranks: list[int],
        now: float,
    ) -> tuple[int, int]:
        if request.prompt_tokens < 0 or (
            request.max_new_tokens is not None and request.max_new_tokens < 0
        ):
            raise DPAdmissionError(400, "Request token counts must be nonnegative.")
        candidates = active_ranks
        if request.routed_dp_rank is not None:
            if request.routed_dp_rank not in active_ranks:
                raise DPAdmissionError(400, "Requested DP rank is not active.")
            candidates = [request.routed_dp_rank]

        known = [
            i
            for i in candidates
            if self.ranks[i].ready
            and now - self.ranks[i].received_at <= self.snapshot_timeout
        ]
        if not known:
            raise DPAdmissionError(
                503, "Decode rank admission capacity is unavailable."
            )

        fitting = []
        for i in known:
            rank = self.ranks[i]
            try:
                tokens = rank.charge(request)
            except DPAdmissionError:
                continue
            if tokens <= rank.token_capacity:
                fitting.append((i, tokens))
        if not fitting:
            if len(known) != len(candidates):
                raise DPAdmissionError(
                    503, "Decode rank admission capacity is unavailable."
                )
            raise DPAdmissionError(
                400,
                "Request prompt, generation budget, and decode reserve exceed "
                "the capacity of every eligible decode rank.",
            )

        available = [
            (i, tokens)
            for i, tokens in fitting
            if self.ranks[i].total_requests < self.ranks[i].request_capacity
            and self.ranks[i].total_tokens + tokens <= self.ranks[i].token_capacity
        ]
        if not available:
            raise DPAdmissionError(
                429, "All eligible decode rank budgets are occupied."
            )
        return min(
            available,
            key=lambda item: (
                self.ranks[item[0]].total_tokens,
                self.ranks[item[0]].total_requests,
                item[0],
            ),
        )

    def reserve(self, rank: int, tokens: int) -> DPAdmissionReservation:
        budget = self.ranks[rank]
        budget.sequence += 1
        budget.pending[budget.sequence] = tokens
        return DPAdmissionReservation(rank, budget.sequence, tokens)

    def rollback_unsent(self, reservation: DPAdmissionReservation) -> None:
        """Only for a failed single-frame IPC send, never for cancellation."""
        self.ranks[reservation.rank].pending.pop(reservation.sequence, None)

    def enqueue(self, request: Any, cost: DPAdmissionRequest, now: float) -> None:
        if len(self.queue) >= self.max_queued_requests:
            raise DPAdmissionError(429, "Decode admission waiting queue is full.")
        self.queue.append(QueuedDPRequest(request, cost, now + self.queue_timeout))

    def abort_queued(self, rid: str, abort_all: bool = False) -> list[Any]:
        if not rid and not abort_all:
            return []
        removed = []
        remaining = deque()
        for entry in self.queue:
            if abort_all or entry.request.rid.startswith(rid):
                removed.append(entry.request)
            else:
                remaining.append(entry)
        self.queue = remaining
        return removed


@dataclass
class _LiveReservation:
    request: Any
    tokens: int
    terminal: bool = False


class SchedulerDPAdmission:
    """A ledger tied to Req identity, including retraction and deferred release."""

    def __init__(
        self,
        epoch: str,
        *,
        page_size: int,
        decode_reserve: int,
        prealloc_reserve: int,
        max_sequence_length: int,
        max_new_tokens_limit: int = 0,
    ):
        self.epoch = epoch
        self.page_size = page_size
        self.decode_reserve = decode_reserve
        self.prealloc_reserve = prealloc_reserve
        self.max_sequence_length = max_sequence_length
        self.max_new_tokens_limit = max_new_tokens_limit or 0
        self.sequence = 0
        self.live: dict[int, _LiveReservation] = {}
        # A monotonic, local change counter. A /v1/loads read may retire an
        # entry before the next shared-memory publication, so the writer must
        # not rely on noticing the removal in its own snapshot() call.
        self.num_released_requests = 0

    def acknowledge(self, tokenized_request: Any) -> None:
        sequence = tokenized_request.dp_admission_sequence
        if (
            tokenized_request.dp_admission_epoch != self.epoch
            or sequence <= self.sequence
            or tokenized_request.dp_admission_tokens <= 0
        ):
            raise RuntimeError("Invalid or out-of-order DP admission reservation.")
        self.sequence = sequence

    def bind(self, request: Any, tokenized_request: Any) -> None:
        # Keep strong references: a completed request may still own KV while a
        # retry is already using its rid. Neither rid nor an output is a refund.
        self.live[id(request)] = _LiveReservation(
            request, tokenized_request.dp_admission_tokens
        )

    def mark_terminal(self, request: Any) -> None:
        entry = self.live.get(id(request))
        if entry is not None and entry.request is request:
            entry.terminal = True

    def snapshot(self, num_used_tokens: int) -> dict:
        reserved_tokens = 0
        allocated_tokens = 0
        for key, entry in list(self.live.items()):
            req = entry.request
            kv = req.kv
            terminal = entry.terminal or bool(req.finished_output)
            owns_resources = (
                kv.holds_kv or kv.holds_mamba or kv.retraction_backup is not None
            )
            if terminal and not owns_resources:
                del self.live[key]
                self.num_released_requests += 1
                continue
            reserved_tokens += entry.tokens
            if kv.holds_kv:
                allocated = (
                    (kv.kv_allocated_len + self.page_size - 1)
                    // self.page_size
                    * self.page_size
                )
                allocated_tokens += min(entry.tokens, allocated)
        # Counting both commitments and all used tokens would count the same
        # preallocated KV twice. Add only allocation not covered by live entries.
        reserved_tokens += max(0, num_used_tokens - allocated_tokens)
        return {
            "dp_admission_epoch": self.epoch,
            "dp_admission_sequence": self.sequence,
            "dp_admission_reserved_tokens": reserved_tokens,
            "dp_admission_reserved_requests": len(self.live),
            "dp_admission_page_size": self.page_size,
            "dp_admission_decode_reserve": self.decode_reserve,
            "dp_admission_prealloc_reserve": self.prealloc_reserve,
            "dp_admission_max_seq_len": self.max_sequence_length,
            "dp_admission_max_new_tokens_limit": self.max_new_tokens_limit,
        }
