"""Bounded observations of SDK results; no keys, payloads, or storage-tier inference."""

import logging
import threading
import time

logger = logging.getLogger(__name__)


class StorageIOObserver:
    """Interval deltas, atomically drained by the metrics consumer.

    Objects are SDK keys, pages are adapter results. A successful put transfers
    the supplied buffer sizes; a get transfers only its positive returned bytes.
    Exists misses are distinct from errors. Short replies count absent results as
    failures without modifying the reply or the adapter's handling of it.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._io = {}
        self._pages = {}
        self._batch_observer = None
        self._batch_observer_error_logged = False

    def set_batch_observer(self, observer):
        self._batch_observer = observer

    def _notify_batch(self, event):
        if self._batch_observer is None:
            return
        try:
            self._batch_observer(event)
        except Exception:
            if not self._batch_observer_error_logged:
                self._batch_observer_error_logged = True
                logger.exception("HiCache storage batch metric failed")

    def call(self, operation, count, sizes, fn, *args, pool="unknown"):
        requested_bytes = sum(
            sum(size) if isinstance(size, (list, tuple)) else size
            for size in (sizes or ())
        )
        event = {
            "operation": operation,
            "pool": str(pool),
            "objects": count,
            "requested_bytes": requested_bytes,
        }
        self._notify_batch(dict(event, phase="start"))
        start = time.perf_counter()
        try:
            result = fn(*args)
        except Exception:
            duration = time.perf_counter() - start
            self._record(operation, count, sizes, (), duration, 1)
            self._notify_batch(
                dict(
                    event,
                    phase="finish",
                    outcome="exception",
                    duration_seconds=duration,
                )
            )
            raise
        duration = time.perf_counter() - start
        values = self._record(operation, count, sizes, result, duration, 0)
        outcome = (
            "success"
            if values["successful_objects"] == count
            else "partial"
            if values["successful_objects"]
            else "failure"
        )
        self._notify_batch(
            dict(event, phase="finish", outcome=outcome, duration_seconds=duration)
        )
        return result

    def _record(self, operation, count, sizes, result, duration, exceptions):
        successful = missing = transferred = 0
        for i, value in enumerate(result[:count]):
            success = (
                value > 0
                if operation == "get"
                else value == (1 if operation == "exists" else 0)
            )
            if success:
                successful += 1
                if operation == "get":
                    transferred += value
                elif operation == "put":
                    size = sizes[i]
                    transferred += (
                        sum(size) if isinstance(size, (list, tuple)) else size
                    )
            elif operation == "exists" and value == 0:
                missing += 1
        values = {
            "calls": 1,
            "duration_seconds": duration,
            "requested_objects": count,
            "successful_objects": successful,
            "failed_objects": count - successful - missing,
            "missing_objects": missing,
            "successful_bytes": transferred,
            "partial_calls": int(0 < successful < count),
            "exceptions": exceptions,
        }
        with self._lock:
            counters = self._io.setdefault(operation, dict.fromkeys(values, 0))
            for name, value in values.items():
                counters[name] += value
        return values

    def pages(self, operation, results):
        count = len(results)
        successful = sum(results)
        values = {
            "requested_pages": count,
            "successful_pages": successful,
            "failed_pages": count - successful,
            "partial_batches": int(0 < successful < count),
        }
        with self._lock:
            counters = self._pages.setdefault(operation, dict.fromkeys(values, 0))
            for name, value in values.items():
                counters[name] += value

    def drain(self):
        with self._lock:
            io, pages = self._io, self._pages
            self._io, self._pages = {}, {}
        return io, pages
