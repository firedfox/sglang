"""Bounded-label HiCache observations; instantiated once by the cache owner."""

DURATION_BUCKETS = (
    0.0001,
    0.0005,
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.5,
    1,
    5,
    10,
    30,
    60,
)


class CacheMetricsCollector:
    def __init__(self, labels, registry=None):
        # Import only after the runtime configures prometheus multiprocessing.
        from prometheus_client import REGISTRY, Counter, Gauge, Histogram

        registry = REGISTRY if registry is None else registry
        self.labels = labels
        base = list(labels)

        def counter(name, doc, extra):
            return Counter("sglang:" + name, doc, base + extra, registry=registry)

        def histogram(name, doc, extra, buckets=DURATION_BUCKETS):
            return Histogram(
                "sglang:" + name, doc, base + extra, buckets=buckets, registry=registry
            )

        self.prefetch = counter(
            "cache_prefetch_tokens",
            "Rank-local prefetch stage token facts; stages are not additive.",
            ["stage"],
        )
        self.loss = counter(
            "cache_loss_tokens",
            "Prefetch stage differences, including deduplication, not all recomputation.",
            ["reason"],
        )
        self.operations = counter(
            "cache_operations",
            "Finalized cache operations by terminal reason.",
            ["operation", "reason"],
        )
        self.duration = histogram(
            "cache_stage_duration_seconds",
            "Local monotonic stage duration; overlapping stages are not additive.",
            ["operation", "stage"],
        )
        self.request = counter(
            "cache_request_tokens",
            "First-match and used-prefix tokens, once per request on each scheduler rank.",
            ["tier", "stage"],
        )
        self.requests = counter(
            "cache_requests",
            "Observed cache request summaries by terminal outcome on each scheduler rank.",
            ["outcome"],
        )
        self.wait = histogram(
            "cache_request_wait_seconds",
            "Observed scheduler scan waiting interval, not causal TTFT overhead.",
            ["stage"],
        )
        self.lookup = histogram(
            "cache_lookup_seconds",
            "Initial prefix lookup CPU duration.",
            [],
            (1e-6, 5e-6, 1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1),
        )
        self.transfer_tokens = counter(
            "cache_transfer_tokens",
            "Completed rank-local GPU/Host transfer tokens by direction and pool.",
            ["direction", "pool"],
        )
        self.transfer_bytes = counter(
            "cache_transfer_bytes",
            "Completed rank-local GPU/Host transfer bytes by direction.",
            ["direction"],
        )
        self.transfer_duration = histogram(
            "cache_transfer_duration_seconds",
            "Completed rank-local GPU/Host transfer duration.",
            ["direction"],
        )
        self.state = Gauge(
            "sglang:cache_state_tokens",
            "Rank-local token slots/budget snapshot; states may overlap.",
            base + ["state"],
            registry=registry,
            multiprocess_mode="mostrecent",
        )
        self.queue = Gauge(
            "sglang:cache_queue_operations",
            "Rank-local queue entries (release queue entries are pages).",
            base + ["queue"],
            registry=registry,
            multiprocess_mode="mostrecent",
        )
        self.io = counter(
            "cache_storage_io",
            "SDK batch I/O observations; put bytes are successful objects' supplied buffer sizes, not confirmed network bytes.",
            ["operation", "field"],
        )
        batch_labels = ["operation", "pool", "object_bucket", "outcome"]
        self.storage_batch_duration = histogram(
            "cache_storage_batch_duration_seconds",
            "Wall time of one Mooncake SDK call, including its metadata lookup and transfer; not prefetch queue time.",
            batch_labels,
            (
                0.01,
                0.02,
                0.05,
                0.1,
                0.2,
                0.3,
                0.5,
                0.75,
                1,
                1.5,
                2,
                3,
                4,
                5,
                6,
                8,
                10,
                15,
                30,
            ),
        )
        self.storage_batch_objects = counter(
            "cache_storage_batch_objects",
            "Objects submitted to Mooncake SDK calls by batch size and result.",
            batch_labels,
        )
        self.storage_batch_requested_bytes = counter(
            "cache_storage_batch_requested_bytes",
            "Supplied buffer sizes for Mooncake SDK calls; not confirmed network bytes.",
            batch_labels,
        )
        self.storage_batch_inflight = Gauge(
            "sglang:cache_storage_batch_inflight",
            "Mooncake SDK calls currently active in this rank process.",
            base + ["operation", "pool"],
            registry=registry,
            multiprocess_mode="livesum",
        )
        self.queue_age = Gauge(
            "sglang:cache_queue_oldest_age_seconds",
            "Age since operation creation for the oldest queued item (not pure queue residence).",
            base + ["queue"],
            registry=registry,
            multiprocess_mode="mostrecent",
        )
        self.protected = Gauge(
            "sglang:cache_protected_tokens",
            "Unique main Host KV logical slots covered by Unified host locks, not allocator unevictable capacity.",
            base + ["tier"],
            registry=registry,
            multiprocess_mode="mostrecent",
        )
        protection_labels = base + ["tier", "reason", "policy"]
        self.protection = Gauge(
            "sglang:cache_protection_tokens",
            "Unified Host KV slots covered by the unified_host_lock summary.",
            protection_labels,
            registry=registry,
            multiprocess_mode="mostrecent",
        )
        self.host_bytes = Gauge(
            "sglang:cache_host_protected_bytes",
            "Unique covered main Host KV payload bytes, including colocated draft but excluding independent sidecars and allocator overhead.",
            base,
            registry=registry,
            multiprocess_mode="mostrecent",
        )
        self.inflight_age = Gauge(
            "sglang:cache_inflight_oldest_age_seconds",
            "Oldest physically unfinished prefetch age since creation, including cancelled operations.",
            base,
            registry=registry,
            multiprocess_mode="mostrecent",
        )
        self.references = Gauge(
            "sglang:cache_protection_references",
            "Active Unified Host lock references.",
            protection_labels,
            registry=registry,
            multiprocess_mode="mostrecent",
        )
        self.pages = counter(
            "cache_storage_pages",
            "Adapter v1/v2 logical page result counts; not prefix hits.",
            ["operation", "field"],
        )

    def observe_operation(self, obs, operation="prefetch"):
        self.operations.labels(
            **self.labels, operation=operation, reason=obs.reason
        ).inc()
        if operation == "prefetch":
            for stage, count in getattr(obs, "final_tokens", obs.tokens).items():
                self.prefetch.labels(**self.labels, stage=stage).inc(max(0, count))
            for reason, count in obs.losses.items():
                self.loss.labels(**self.labels, reason=reason).inc(count)
        durations = obs.backup_durations() if operation == "backup" else obs.durations()
        for stage, seconds in durations.items():
            if operation == "prefetch" and stage == "get":
                continue  # Physical completion reaper owns GET duration exactly once.
            self.duration.labels(
                **self.labels, operation=operation, stage=stage
            ).observe(seconds)

    def observe_request(self, data):
        self.requests.labels(**self.labels, outcome=data["outcome"]).inc()
        for stage in ("matched", "used"):
            for tier, count in data[stage].items():
                self.request.labels(**self.labels, tier=tier, stage=stage).inc(
                    max(0, count)
                )
        self.lookup.labels(**self.labels).observe(data["lookup_ms"] / 1000)
        for stage, ms in data["wait_ms"].items():
            self.wait.labels(**self.labels, stage=stage).observe(ms / 1000)

    def observe_transfer(self, direction, tokens_by_pool, num_bytes, duration_seconds):
        for pool, tokens in (tokens_by_pool or {}).items():
            if tokens > 0:
                self.transfer_tokens.labels(
                    **self.labels, direction=direction, pool=pool
                ).inc(tokens)
        if num_bytes > 0:
            self.transfer_bytes.labels(**self.labels, direction=direction).inc(
                num_bytes
            )
        if duration_seconds is not None:
            self.transfer_duration.labels(**self.labels, direction=direction).observe(
                duration_seconds
            )

    def observe_storage(self, stats):
        for operation, fields in getattr(stats, "io_stats", {}).items():
            for field, value in fields.items():
                self.io.labels(**self.labels, operation=operation, field=field).inc(
                    value
                )
        for operation, fields in getattr(stats, "page_stats", {}).items():
            for field, value in fields.items():
                self.pages.labels(**self.labels, operation=operation, field=field).inc(
                    value
                )

    def observe_storage_batch(self, event):
        operation = event["operation"]
        pool = event["pool"]
        inflight = self.storage_batch_inflight.labels(
            **self.labels, operation=operation, pool=pool
        )
        if event["phase"] == "start":
            inflight.inc()
            return
        inflight.dec()
        count = event["objects"]
        bucket = next(
            (
                str(limit)
                for limit in (128, 256, 512, 1024, 1536, 2048, 4096)
                if count <= limit
            ),
            ">4096",
        )
        labels = dict(
            self.labels,
            operation=operation,
            pool=pool,
            object_bucket=bucket,
            outcome=event["outcome"],
        )
        self.storage_batch_duration.labels(**labels).observe(event["duration_seconds"])
        self.storage_batch_objects.labels(**labels).inc(count)
        self.storage_batch_requested_bytes.labels(**labels).inc(
            event["requested_bytes"]
        )

    def snapshot_unified_host(self, tokens, bytes_per_token, references, policy):
        """Publish Full Host slots retained by UnifiedRadixCache host locks."""
        labels = dict(
            self.labels, tier="host", reason="unified_host_lock", policy=policy
        )
        self.protected.labels(**self.labels, tier="host").set(tokens)
        self.protection.labels(**labels).set(tokens)
        self.references.labels(**labels).set(references)
        self.host_bytes.labels(**self.labels).set(tokens * bytes_per_token)

    def snapshot_unified_runtime(self, cache):
        """Publish rank-local Unified capacity and async pipeline pressure."""
        import time

        cc = cache.cache_controller
        host = cc.mem_pool_host
        inflight = cache._cache_prefetch_io.values()
        release_queue = getattr(cc, "host_mem_release_queue", None)
        states = {
            "host_capacity": host.size,
            "host_free": host.available_size(),
            "device_protected": cache.protected_size(),
            "prefetch_budget_used": getattr(cc, "prefetch_tokens_occupied", 0),
            "prefetch_budget_limit": getattr(cc, "prefetch_capacity_limit", 0),
            "prefetch_allocated_uncommitted": sum(
                op.cache_observation.tokens.get("allocated", 0)
                for op in inflight
                if not op.cache_observation.finalized
            ),
            "prefetch_cancelled_io_inflight": sum(
                op.cache_observation.tokens.get("allocated", 0)
                for op in inflight
                if op.cache_observation.finalized
                and "get_done" not in op.cache_observation.times
            ),
            "host_release_pending": (
                release_queue.qsize() * cc.page_size if release_queue is not None else 0
            ),
        }
        for state, value in states.items():
            self.state.labels(**self.labels, state=state).set(value)
        self.inflight_age.labels(**self.labels).set(
            max(
                (
                    time.monotonic() - op.cache_observation.times["created"]
                    for op in inflight
                    if "get_done" not in op.cache_observation.times
                ),
                default=0.0,
            )
        )
        for name in (
            "prefetch_queue",
            "prefetch_hit_queue",
            "prefetch_buffer",
            "backup_queue",
            "ack_backup_queue",
            "host_mem_release_queue",
        ):
            queue = getattr(cc, name, None)
            self.queue.labels(**self.labels, queue=name).set(
                queue.qsize() if queue is not None else 0
            )
            age = 0.0
            if queue is not None and name != "host_mem_release_queue":
                with queue.mutex:
                    first = queue.queue[0] if queue.queue else None
                obs = getattr(first, "cache_observation", None)
                if obs is not None:
                    age = max(0.0, time.monotonic() - obs.times["created"])
            self.queue_age.labels(**self.labels, queue=name).set(age)
