"""CPU regression tests for serving benchmark accounting and native SSE.

Exercise the real HTTP request function and benchmark orchestration. No model,
GPU, tokenizer download, or external service is needed.
"""

import asyncio
import json
import tempfile
import threading
import unittest
import warnings
from argparse import Namespace
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import requests
from sglang.benchmark import serving
from sglang.benchmark.datasets import DatasetRow
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def _chunk(tokens, text="hello", **meta):
    return {
        "text": text,
        "meta_info": {"prompt_tokens": 8, "completion_tokens": tokens, **meta},
    }


@contextmanager
def _native_server(reply):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with self.server.lock:
                self.server.bodies.append(body)
            response = reply(body) if callable(reply) else reply
            status = response.get("status", 200)
            self.send_response(status)
            if "json" in response:
                self.send_header("Content-Type", "application/json")
                payload = json.dumps(response["json"], indent=2).encode()
            else:
                self.send_header("Content-Type", "text/event-stream")
                payload = b""
                for chunk in response.get("chunks", []):
                    payload += (
                        chunk
                        if isinstance(chunk, bytes)
                        else b"data: " + json.dumps(chunk).encode() + b"\n\n"
                    )
                if response.get("done", True):
                    payload += b"data: [DONE]\n\n"
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                b'{"decode":[{"internal_states":[{"avg_spec_accept_length":9.0}]}]}'
            )

        def log_message(self, fmt, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.bodies = []
    server.lock = threading.Lock()
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class TestServingMeasurement(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output_file = Path(self.temp.name) / "result.jsonl"
        self.args = Namespace(
            backend="sglang",
            dataset_name="random",
            temperature=0.0,
            top_p=1.0,
            disable_ignore_eos=False,
            disable_stream=False,
            return_logprob=False,
            return_routed_experts=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            header=None,
            cache_report=True,
            warmup_requests=0,
            plot_throughput=False,
            goodput=None,
            profile_steps=None,
            profile_num_steps=None,
            sharegpt_output_len=None,
            random_input_len=8,
            random_output_len=16,
            random_range_ratio=1.0,
            output_file=str(self.output_file),
            output_details=True,
        )
        self.args_patch = patch.object(serving, "args", self.args, create=True)
        self.args_patch.start()
        self.addCleanup(self.args_patch.stop)

    def _request(self, base_url, output_len=16):
        return serving.RequestFuncInput(
            prompt="hello",
            api_url=base_url + "/generate",
            prompt_len=8,
            output_len=output_len,
            model="test-model",
            lora_name="",
            image_data=None,
            extra_request_body={},
        )

    def _run_request(self, reply, output_len=16):
        with _native_server(reply) as (server, url):
            result = asyncio.run(
                serving.async_request_sglang_generate(self._request(url, output_len))
            )
            self.assertEqual(result.http_status, reply.get("status", 200))
            self.assertGreater(result.end_time, result.start_time)
            self.assertAlmostEqual(result.latency, result.end_time - result.start_time)
            return result, server.bodies

    def _benchmark(self, base_url, count=2, profile=False):
        return asyncio.run(
            serving.benchmark(
                backend="sglang",
                api_url=base_url + "/generate",
                base_url=base_url,
                model_id="test-model",
                tokenizer=_Tokenizer(),
                input_requests=[
                    DatasetRow(prompt=f"request {i}", prompt_len=8, output_len=16)
                    for i in range(count)
                ],
                request_rate=float("inf"),
                max_concurrency=2,
                disable_tqdm=True,
                lora_names=None,
                lora_request_distribution=None,
                lora_zipf_alpha=None,
                extra_request_body={},
                profile=profile,
                pd_separated=profile,
                profile_prefill_url=[base_url] if profile else None,
                warmup_requests=0,
            )
        )

    def _saved_result(self):
        def invalid_constant(value):
            raise AssertionError(f"Non-JSON numeric constant: {value}")

        lines = self.output_file.read_text().splitlines()
        self.assertEqual(len(lines), 1)
        return json.loads(lines[0], parse_constant=invalid_constant)

    def test_normal_legacy_native_stream_with_done_without_finish_reason(self):
        result, bodies = self._run_request(
            {"chunks": [_chunk(1), _chunk(3, "hello there now")], "done": True}
        )
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output_len, 3)
        self.assertEqual(result.completion_tokens, 3)
        self.assertEqual(len(result.itl), 2)
        self.assertGreater(result.ttft, 0)
        self.assertEqual(bodies[0]["sampling_params"]["max_new_tokens"], 16)

    def test_native_normal_finish_reason_is_sufficient_without_done(self):
        for reason in ({"type": "stop"}, {"type": "length", "length": 3}):
            with self.subTest(reason=reason):
                result, _ = self._run_request(
                    {
                        "chunks": [_chunk(1), _chunk(3, finish_reason=reason)],
                        "done": False,
                    }
                )
                self.assertTrue(result.success, result.error)
                self.assertEqual(result.output_len, 3)
                self.assertEqual(result.finish_reason, reason)

    def test_native_partial_eof_without_terminal_marker_is_not_successful(self):
        result, _ = self._run_request(
            {"chunks": [_chunk(1), _chunk(2, "hello partial")], "done": False}
        )
        self.assertFalse(result.success)
        self.assertEqual(result.output_len, 2)
        self.assertEqual(result.completion_tokens, 2)
        self.assertEqual(result.generated_text, "hello partial")
        self.assertIn("without a terminal finish_reason or [DONE]", result.error)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics, lengths = serving.calculate_metrics(
                None, [result], 1, _Tokenizer(), "sglang"
            )
        self.assertEqual(metrics.completed, 0)
        self.assertEqual(metrics.total_output, 0)
        self.assertEqual(metrics.output_throughput, 0)
        self.assertEqual(lengths, [0])

    def test_done_empty_and_error_streams_are_not_successes(self):
        cases = (
            {"chunks": []},
            {"chunks": [], "done": False},
            {"chunks": [{"error": {"message": "upstream failed"}}]},
            {"chunks": [{"text": "text without actual token metadata"}]},
        )
        for reply in cases:
            with self.subTest(reply=reply):
                result, _ = self._run_request(reply, output_len=1024)
                self.assertFalse(result.success)
                self.assertEqual(result.output_len, 0)
                self.assertIsNone(result.completion_tokens)
                self.assertTrue(result.error)

    def test_partial_stream_abort_keeps_actual_count_but_fails(self):
        reason = {"type": "abort", "message": "cancelled", "status_code": 503}
        result, _ = self._run_request(
            {"chunks": [_chunk(2), _chunk(2, finish_reason=reason)]}
        )
        self.assertFalse(result.success)
        self.assertEqual(result.completion_tokens, 2)
        self.assertEqual(result.finish_reason, reason)
        self.assertIn("cancelled", result.error)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics, lengths = serving.calculate_metrics(
                None, [result], 1, _Tokenizer(), "sglang"
            )
        self.assertEqual(metrics.total_output, 0)
        self.assertEqual(lengths, [0])

    def test_final_empty_text_metadata_updates_counts_and_preserves_cache(self):
        cache = {"device": 4, "host": 2, "storage": 0}
        result, _ = self._run_request(
            {
                "chunks": [
                    _chunk(1, cached_tokens=6, cached_tokens_details=cache),
                    _chunk(
                        3,
                        text="",
                        finish_reason={"type": "length", "length": 3},
                        spec_verify_ct=1,
                        spec_num_correct_drafts=2,
                        spec_num_proposed_drafts=3,
                    ),
                ]
            }
        )
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output_len, 3)
        self.assertEqual(result.generated_text, "hello")
        self.assertEqual(result.cached_tokens, 6)
        self.assertEqual(result.cached_tokens_details, cache)
        self.assertEqual(result.spec_verify_ct, 1)

    def test_non_stream_and_explicit_zero_tokens_use_server_counts(self):
        self.args.disable_stream = True
        for tokens, reason in (
            (0, {"type": "length", "length": 0}),
            (3, {"type": "length", "length": 3}),
            (3, None),
        ):
            with self.subTest(tokens=tokens, reason=reason):
                result, bodies = self._run_request(
                    {
                        "json": _chunk(
                            tokens,
                            text="" if tokens == 0 else "hello",
                            finish_reason=reason,
                        )
                    }
                )
                self.assertTrue(result.success, result.error)
                self.assertEqual(result.completion_tokens, tokens)
                self.assertEqual(result.output_len, tokens)
                self.assertFalse(bodies[0]["stream"])

    def test_http_rejection_preserves_status_and_elapsed_time(self):
        for status in (429, 503):
            with self.subTest(status=status):
                result, _ = self._run_request(
                    {"status": status, "json": {"error": "capacity unavailable"}}
                )
                self.assertFalse(result.success)
                self.assertEqual(result.output_len, 0)
                self.assertIn("capacity unavailable", result.error)

    def test_exact_peak_does_not_sum_distinct_requests_in_a_second(self):
        def output(start, end, success=True):
            return serving.RequestFuncOutput(
                success=success,
                start_time=start,
                end_time=end,
                latency=end - start,
                output_len=2,
                ttft=0.01,
                itl=[0.01],
            )

        cases = (
            ([output(1, 1.1), output(1.2, 1.3)], 1),
            ([output(1, 2), output(2, 3)], 1),
            ([output(1, 2), output(1.1, 1.2, success=False)], 2),
        )
        for outputs, peak in cases:
            with self.subTest(peak=peak, outputs=outputs):
                metrics, _ = serving.calculate_metrics(
                    None, outputs, 3, _Tokenizer(), "sglang"
                )
                self.assertEqual(metrics.max_concurrent_requests, peak)
                self.assertAlmostEqual(
                    metrics.inflight_concurrency,
                    sum(out.end_time - out.start_time for out in outputs) / 3,
                )

    def test_zero_output_does_not_invent_a_first_token_or_peak_token_rate(self):
        output = serving.RequestFuncOutput(
            success=True, start_time=1, end_time=2, latency=1, output_len=0
        )
        metrics, _ = serving.calculate_metrics(
            None, [output], 1, _Tokenizer(), "sglang"
        )
        self.assertEqual(metrics.total_output, 0)
        self.assertEqual(metrics.max_output_tokens_per_s, 0)
        self.assertIsNone(metrics.mean_ttft_ms)
        self.assertIsNone(metrics.mean_tpot_ms)

    def test_rate_limited_generator_has_no_idle_interval_after_last_request(self):
        requests = [DatasetRow(prompt="hi", prompt_len=1, output_len=1)] * 3

        async def collect():
            return [row async for row in serving.get_request(requests, 1)]

        with (
            patch.object(serving.asyncio, "sleep", new_callable=AsyncMock) as sleep,
            patch.object(serving.np.random, "exponential", return_value=2),
        ):
            actual = asyncio.run(collect())
        self.assertEqual(actual, requests)
        self.assertEqual(sleep.await_count, len(requests) - 1)

    def test_all_failed_benchmark_writes_auditable_json_even_if_metadata_fails(self):
        with (
            _native_server({"status": 429, "json": {"error": "full"}}) as (_, url),
            patch.object(
                serving.requests,
                "get",
                side_effect=requests.ConnectionError("server_info unavailable"),
            ),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            result = self._benchmark(url)
        saved = self._saved_result()
        self.assertEqual(saved["measurement_schema_version"], 2)
        self.assertEqual(saved["offered_requests"], 2)
        self.assertEqual(saved["completed"], 0)
        self.assertEqual(saved["failed_requests"], 2)
        self.assertEqual(saved["http_status_counts"], {"429": 2})
        self.assertEqual(sum(saved["error_counts"].values()), 2)
        self.assertEqual(saved["total_output_tokens"], 0)
        self.assertEqual(saved["output_throughput"], 0)
        self.assertIsNone(saved["server_info"])
        for name in (
            "mean_e2e_latency_ms",
            "p99_ttft_ms",
            "mean_itl_ms",
            "mean_tpot_ms",
        ):
            self.assertIsNone(saved[name])
        self.assertEqual(result["request_details"], saved["request_details"])
        for detail in saved["request_details"]:
            self.assertFalse(detail["success"])
            self.assertEqual(detail["http_status"], 429)
            self.assertGreater(detail["end_time"], detail["start_time"])
            self.assertGreater(detail["latency"], 0)
            self.assertIsNone(detail["completion_tokens"])
            self.assertTrue(detail["error"])

    def test_failure_summary_survives_without_output_details(self):
        self.args.output_details = False
        with (
            _native_server({"status": 503, "json": {"error": "overloaded"}}) as (
                _,
                url,
            ),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            self._benchmark(url)
        saved = self._saved_result()
        self.assertNotIn("request_details", saved)
        self.assertEqual(saved["http_status_counts"], {"503": 2})
        self.assertEqual(sum(saved["error_counts"].values()), 2)
        self.assertIn("overloaded", next(iter(saved["error_counts"])))

    def test_workload_clock_excludes_profiler_stop_and_metadata(self):
        class Clock:
            value = 100.0

            def now(self):
                self.value += 0.001
                return self.value

        clock = Clock()

        async def profile(urls, mode):
            if mode == "stop":
                clock.value += 20

        def metadata(url):
            clock.value += 40
            return {}

        with (
            _native_server({"chunks": [_chunk(2)]}) as (_, url),
            patch.object(serving, "time", SimpleNamespace(perf_counter=clock.now)),
            patch.object(serving, "_call_profile_pd", side_effect=profile),
            patch.object(serving, "_get_benchmark_server_info", side_effect=metadata),
        ):
            result = self._benchmark(url, count=1, profile=True)
        self.assertGreater(result["duration"], 0)
        self.assertLess(result["duration"], 1)
        self.assertGreaterEqual(result["metadata_collection_duration"], 40)
        self.assertAlmostEqual(
            result["duration"],
            result["workload_end_time"] - result["workload_start_time"],
        )
        self.assertAlmostEqual(result["output_throughput"], 2 / result["duration"])

    def test_request_counters_are_weighted_and_do_not_use_lifetime_first_rank(self):
        def reply(body):
            verify, correct, proposed = (
                (1, 5, 5) if body["text"].endswith("0") else (3, 3, 9)
            )
            return {
                "chunks": [
                    _chunk(
                        6,
                        spec_verify_ct=verify,
                        spec_num_correct_drafts=correct,
                        spec_num_proposed_drafts=proposed,
                    )
                ]
            }

        with _native_server(reply) as (_, url):
            result = self._benchmark(url)
        measured = result["request_speculative_metrics"]
        self.assertTrue(measured["complete"])
        self.assertEqual(measured["requests_with_verify_counters"], 2)
        self.assertEqual(measured["accept_length"], 12 / 4)
        self.assertEqual(measured["accept_rate"], 8 / 14)
        self.assertEqual(result["accept_length"], 9)
        self.assertEqual(
            result["accept_length_source"], "first_decode_worker_first_rank_lifetime"
        )

    def test_missing_verify_counters_do_not_claim_whole_run_acceptance(self):
        outputs = [
            serving.RequestFuncOutput(
                success=True, completion_tokens=6, spec_verify_ct=2
            ),
            serving.RequestFuncOutput(success=True, completion_tokens=6),
        ]
        summary = serving._request_speculative_metrics(outputs)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["requests_with_verify_counters"], 1)
        self.assertIsNone(summary["accept_length"])
        self.assertIsNone(summary["verify_count"])


if __name__ == "__main__":
    unittest.main()
