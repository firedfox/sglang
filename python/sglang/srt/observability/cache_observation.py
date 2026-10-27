"""Cache facts shared by metrics and compact logs; no runtime/GPU dependencies."""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field


def cache_summary_json(data):
    return json.dumps(data, separators=(",", ":"), ensure_ascii=True)


class SlowBackupMonitor:
    """Scheduler-side, bounded snapshots; never wait on storage I/O.

    Queue heads and the active worker operation cover queue/PUT/ACK independently.
    Snapshots are approximate across threads, not an atomic transfer ledger.
    """

    def __init__(self, threshold=3.0, interval=30.0, clock=time.monotonic):
        self.threshold = threshold
        self.interval = max(1.0, interval)
        self.clock = clock
        self.next_check = 0.0
        self.next_log = 0.0

    def poll(self, controller):
        now = self.clock()
        if self.threshold <= 0 or now < max(self.next_check, self.next_log):
            return None
        self.next_check = now + 1.0
        candidates = []
        sizes = {}
        for name in ("backup_queue", "ack_backup_queue"):
            queue = getattr(controller, name, None)
            first = None
            sizes[name] = 0
            if queue is not None:
                with queue.mutex:
                    sizes[name] = len(queue.queue)
                    first = queue.queue[0] if queue.queue else None
            candidates.append(first)
        candidates.insert(1, getattr(controller, "active_backup_operation", None))
        rows, seen = [], set()
        for operation in candidates:
            if operation is None or operation.id in seen:
                continue
            seen.add(operation.id)
            obs = operation.cache_observation
            times = dict(obs.times)
            age = max(0.0, now - times["created"])
            if age < self.threshold:
                continue
            phase, started = "queue", times["created"]
            if "get_done" in times:
                phase, started = "ack", times["get_done"]
            elif "get_start" in times:
                phase, started = "put", times["get_start"]
            rows.append(
                dict(
                    operation_id=operation.id,
                    phase=phase,
                    tokens=obs.requested,
                    completed_tokens=operation.completed_tokens,
                    age_seconds=round(age, 3),
                    phase_seconds=round(max(0.0, now - started), 3),
                )
            )
        if not rows:
            return None
        self.next_log = now + self.interval
        worker = getattr(controller, "backup_thread", None)
        return dict(
            policy=controller.write_policy,
            worker_alive=worker.is_alive() if worker is not None else False,
            queues=sizes,
            operations=rows,
        )


@dataclass
class CacheOperationObservation:
    requested: int
    clock: Callable = field(default=time.monotonic, repr=False)
    tokens: dict = field(default_factory=dict)
    times: dict = field(default_factory=dict)
    losses: dict = field(default_factory=dict)
    reason: str = "complete"
    finalized: bool = False
    pressure: dict = field(default_factory=dict)

    def __post_init__(self):
        self.tokens["requested"] = self.requested
        self.mark("created")

    def mark(self, stage):
        self.times.setdefault(stage, self.clock())

    def durations(self):
        pairs = {
            "query_queue": ("created", "query_start"),
            "query": ("query_start", "query_done"),
            "query_sync": ("query_done", "query_synced"),
            "allocation_wait": ("query_synced", "allocated"),
            "get_queue": ("allocated", "get_start"),
            "get": ("get_start", "get_done"),
            "finalize_wait": ("get_done", "finalized"),
            "prefetch": ("created", "finalized"),
        }
        return {
            stage: self.times[b] - self.times[a]
            for stage, (a, b) in pairs.items()
            if a in self.times and b in self.times and self.times[b] >= self.times[a]
        }

    def backup_durations(self):
        pairs = {
            "queue": ("created", "get_start"),
            "io": ("get_start", "get_done"),
            "ack_wait": ("get_done", "finalized"),
            "total": ("created", "finalized"),
        }
        return {
            stage: self.times[b] - self.times[a]
            for stage, (a, b) in pairs.items()
            if a in self.times and b in self.times and self.times[b] >= self.times[a]
        }

    def finish(self, local, synced, duplicate, reason="complete"):
        if self.finalized:
            return False
        self.tokens.update(
            completed_local=local,
            completed=synced,
            duplicate=duplicate,
            inserted=max(0, synced - duplicate),
        )
        self.reason = reason
        self.mark("finalized")
        self.finalized = True
        self.final_tokens = dict(self.tokens)
        query = self.final_tokens.get("query", 0)
        allocated = self.tokens.get("allocated", 0)
        differences = [
            (
                "query_rank_min",
                self.final_tokens.get("query_local", query) - query
                if "query" in self.final_tokens
                else 0,
            ),
            ("host_allocation_truncated" if allocated else reason, query - allocated),
            (reason if reason != "complete" else "get_partial", allocated - local),
            (
                "cancelled" if reason == "cancelled" else "completion_rank_or_pool_min",
                local - synced,
            ),
            ("duplicate", duplicate),
        ]
        for cause, count in differences:
            if count > 0:
                self.losses[cause] = self.losses.get(cause, 0) + count
        return True

    def snapshot(self):
        return {
            "tokens": dict(getattr(self, "final_tokens", self.tokens)),
            "reason": self.reason,
            "losses": dict(self.losses),
            "seconds": self.durations(),
            "pressure": dict(self.pressure),
            "source": "unknown",
            "operation_id": getattr(self, "operation_id", None),
        }


class CacheNodeMutationHistory:
    """Bounded, process-local node transitions for enqueue/admission correlation."""

    def __init__(self, max_nodes=8192, events_per_node=4, clock=time.monotonic):
        self.max_nodes = max_nodes
        self.events_per_node = events_per_node
        self.clock = clock
        self.events = OrderedDict()
        self.lock = threading.Lock()
        self.sequence = 0
        self.generation = 0
        self.discarded_through = 0
        self._eviction_context = threading.local()

    @contextmanager
    def eviction_context(
        self,
        *,
        caller,
        allocation_tokens,
        requested_evict_tokens,
        host_free_before,
        component,
        trigger_rid=None,
        trigger_node_id=None,
    ):
        previous = getattr(self._eviction_context, "value", None)
        self._eviction_context.value = dict(
            caller=caller,
            allocation_tokens=allocation_tokens,
            requested_evict_tokens=requested_evict_tokens,
            host_free_before=host_free_before,
            component=component,
        )
        if trigger_rid is not None:
            self._eviction_context.value["trigger_rid"] = trigger_rid
        if trigger_node_id is not None:
            self._eviction_context.value["trigger_node_id"] = trigger_node_id
        try:
            yield
        finally:
            self._eviction_context.value = previous

    def reset(self):
        with self.lock:
            self.events.clear()
            self.generation += 1
            self.discarded_through = self.sequence

    def checkpoint(self):
        with self.lock:
            return (self.generation, self.sequence, self.clock())

    def record(self, node_id, action, tokens):
        with self.lock:
            self.sequence += 1
            if node_id not in self.events:
                if len(self.events) >= self.max_nodes:
                    _, discarded = self.events.popitem(last=False)
                    self.discarded_through = max(
                        self.discarded_through, discarded[-1]["sequence"]
                    )
                self.events[node_id] = deque(maxlen=self.events_per_node)
            else:
                self.events.move_to_end(node_id)
                if len(self.events[node_id]) == self.events_per_node:
                    self.discarded_through = max(
                        self.discarded_through, self.events[node_id][0]["sequence"]
                    )
            event = dict(
                sequence=self.sequence, action=action, tokens=tokens, at=self.clock()
            )
            if action in ("host_leaf_evicted", "host_duplicate_reclaimed"):
                context = getattr(self._eviction_context, "value", None)
                if context is not None:
                    event.update(context)
            self.events[node_id].append(event)

    def since(self, checkpoint, node_ids):
        generation, sequence, started = checkpoint
        with self.lock:
            if generation != self.generation:
                return {"status": "tree_reset", "events": []}
            if not node_ids:
                return {"status": "node_id_unavailable", "events": []}
            events = []
            for tier, node_id in node_ids.items():
                for event in self.events.get(node_id, ()):
                    if event["sequence"] > sequence:
                        entry = dict(
                            tier=tier,
                            node_id=node_id,
                            action=event["action"],
                            tokens=event["tokens"],
                            after_enqueue_ms=round((event["at"] - started) * 1000, 3),
                        )
                        if "caller" in event:
                            entry.update(
                                sequence=event["sequence"],
                                caller=event["caller"],
                                allocation_tokens=event["allocation_tokens"],
                                requested_evict_tokens=event["requested_evict_tokens"],
                                host_free_before=event["host_free_before"],
                                component=event["component"],
                            )
                            for key in ("trigger_rid", "trigger_node_id"):
                                if key in event:
                                    entry[key] = event[key]
                        events.append(entry)
            truncated = self.discarded_through > sequence
        return {
            "status": (
                "history_truncated"
                if truncated
                else "recorded"
                if events
                else "no_recorded_mutation"
            ),
            "events": events,
        }


@dataclass
class CacheRequestObservation:
    matched: dict = field(default_factory=dict)
    enqueue_match: dict = field(default_factory=dict)
    admission_match: dict = field(default_factory=dict)
    node_mutations: dict = field(default_factory=dict)
    enqueue_node_ids: dict = field(default_factory=dict)
    mutation_checkpoint: tuple | None = field(default=None, repr=False)
    mutation_history: CacheNodeMutationHistory | None = field(default=None, repr=False)
    admission_attempts: int = 0
    last_admission_result: str | None = None
    l3_prefetch_decision: dict = field(default_factory=dict)
    l3_prefetch_attempts: int = 0
    admission_recheck_attempts: int = 0
    admission_recheck_prior_loss_tokens: int = 0
    lookup_seconds: float = 0.0
    prefetch: CacheOperationObservation | None = None
    wait_started: dict = field(default_factory=dict)
    wait_seconds: dict = field(default_factory=dict)
    restore_reasons: dict = field(default_factory=dict)
    emitted: bool = False
    collector: object = field(default=None, repr=False)
    labels: dict = field(default_factory=dict)

    def match(self, device, host, seconds):
        if not self.matched:
            self.matched = {"l1": device, "l2": host}
            self.lookup_seconds = seconds

    def record_enqueue_match(self, device, host, *, node_ids=None, history=None):
        if not self.enqueue_match:
            self.enqueue_match = {"l1": device, "l2": host}
            self.enqueue_node_ids = dict(node_ids or {})
            self.mutation_history = history
            if history is not None:
                self.mutation_checkpoint = history.checkpoint()

    def admission_node_mutations(self, device, host):
        if not self.enqueue_match or device + host >= sum(self.enqueue_match.values()):
            return {"status": "no_prefix_loss", "events": []}
        if self.mutation_history is None or self.mutation_checkpoint is None:
            return {"status": "unsupported", "events": []}
        return self.mutation_history.since(
            self.mutation_checkpoint, self.enqueue_node_ids
        )

    def record_admission_match(
        self, device, host, admitted, result, node_mutations=None
    ):
        self.admission_attempts += 1
        self.last_admission_result = result
        if admitted:
            self.admission_match = {"l1": device, "l2": host}
            self.node_mutations = node_mutations or self.admission_node_mutations(
                device, host
            )

    def record_l3_prefetch_decision(
        self, *, anchor_eligible, remaining_tokens, pending_tokens, operation_started
    ):
        self.l3_prefetch_attempts += 1
        self.l3_prefetch_decision = dict(
            anchor_eligible=anchor_eligible,
            remaining_tokens=remaining_tokens,
            pending_tokens=pending_tokens,
            operation_started=operation_started,
        )

    def record_l3_admission_recheck(self, lost_tokens):
        self.admission_recheck_attempts += 1
        self.admission_recheck_prior_loss_tokens = max(
            self.admission_recheck_prior_loss_tokens, lost_tokens
        )

    def wait_start(self, stage, now):
        self.wait_started.setdefault(stage, now)

    def wait_end(self, stage, now):
        start = self.wait_started.pop(stage, None)
        if start is not None:
            self.wait_seconds[stage] = self.wait_seconds.get(stage, 0.0) + max(
                0.0, now - start
            )

    def result(self, rid, used, outcome="used"):
        if self.emitted:
            return None
        self.emitted = True
        data = dict(
            schema_version=1,
            rid=rid,
            outcome=outcome,
            matched=dict(self.matched),
            enqueue_match=dict(self.enqueue_match),
            admission_match=dict(self.admission_match),
            cache_node_mutations=dict(self.node_mutations),
            admission_attempts=self.admission_attempts,
            last_admission_result=self.last_admission_result,
            l3_prefetch_decision=dict(self.l3_prefetch_decision),
            l3_prefetch_attempts=self.l3_prefetch_attempts,
            used=dict(zip(("l1", "l2", "l3"), used)),
            lookup_ms=round(self.lookup_seconds * 1000, 3),
            wait_ms={k: round(v * 1000, 3) for k, v in self.wait_seconds.items()},
            restore_reasons=dict(self.restore_reasons),
            **self.labels,
        )
        # On a completed forward, used is the prefix proven GPU-ready. An abort
        # before output has no proven used prefix and records zeros instead.
        data["acquired_device"] = dict(data["used"])
        data["l3"] = (
            self.prefetch.snapshot()
            if self.prefetch
            else {"status": "not_queried", "source": "unknown"}
        )
        if (
            outcome == "aborted_before_output"
            and self.prefetch
            and not self.prefetch.finalized
        ):
            data["l3"]["status"] = "pending_at_abort"
            data["l3"]["reason"] = "pending_at_abort"
        if self.prefetch and "query" in data["l3"]["tokens"]:
            data["matched"]["l3"] = data["l3"]["tokens"]["query"]
        data["scope"] = "prefill"
        return data


def emit_cache_summary(req, logger, outcome_override=None):
    """Settle a completed output or an explicitly aborted, unreported request."""
    obs = getattr(req, "cache_observation", None)
    if obs is None or obs.emitted:
        return
    if outcome_override == "aborted_before_output":
        used = (0, 0, 0)
        executed = False
    else:
        if not getattr(req, "_cache_breakdown_computed", False) and not req.finished():
            return
        executed = (
            bool(getattr(req, "output_ids", None))
            or getattr(req, "embedding", None) is not None
        )
        used = (
            (
                req.cached_tokens_device,
                req.cached_tokens_host,
                req.cached_tokens_storage,
            )
            if executed
            else (0, 0, 0)
        )
    for stage in tuple(obs.wait_started):
        obs.wait_end(stage, time.monotonic())
    outcome = outcome_override or ("used" if executed else "not_executed")
    data = obs.result(req.rid, used, outcome)
    if data is None:
        return
    data["retraction_count"] = getattr(req, "retraction_count", 0)
    data["retracted_stain"] = getattr(req, "retracted_stain", False)
    data["cache_breakdown_computed"] = getattr(req, "_cache_breakdown_computed", False)
    enqueue = data["enqueue_match"]
    admission = data["admission_match"]
    lost = (
        max(
            0,
            enqueue.get("l1", 0)
            + enqueue.get("l2", 0)
            - admission.get("l1", 0)
            - admission.get("l2", 0),
        )
        if admission
        else 0
    )
    data["l3_admission_recheck"] = {
        "attempted": obs.admission_recheck_attempts > 0,
        "prior_local_prefix_loss_tokens": (
            obs.admission_recheck_prior_loss_tokens
            if obs.admission_recheck_attempts
            else lost
        ),
        "reason": (
            "reissued_on_match_loss"
            if obs.admission_recheck_attempts
            else "not_admitted"
            if not admission
            else "no_admission_recheck"
            if lost
            else "no_prior_local_prefix_loss"
        ),
    }
    if obs.collector is not None:
        obs.collector.observe_request(data)
    # Stable per-request sampling, without random state on the scheduler path.
    import zlib

    rate = getattr(obs, "log_sample_rate", 0.01)
    sampled = zlib.crc32(str(req.rid).encode()) / 2**32 < rate
    anomalous = (
        bool(
            obs.prefetch and obs.prefetch.reason not in ("complete", "below_threshold")
        )
        or obs.admission_recheck_attempts > 0
        or obs.wait_seconds.get("capacity", 0.0) > 0
        or (outcome == "used" and lost > 0 and sum(used) == 0)
    )
    now = time.monotonic()
    # Bound exception summaries per process too; all outcomes still enter metrics.
    last = getattr(emit_cache_summary, "_last_anomaly", 0.0)
    if anomalous and now - last >= 1.0:
        emit_cache_summary._last_anomaly = now
        sampled = True
    if sampled and getattr(obs, "log_owner", obs.labels.get("tp_rank", 0) == 0):
        logger.info("HiCache request_cache %s", cache_summary_json(data))
