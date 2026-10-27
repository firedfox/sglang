"""CPU regression tests for serving benchmark accounting and native/chat SSE.

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
from unittest.mock import AsyncMock, MagicMock, patch

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


def _chat_delta(content=None, *, reasoning=None, finish=None, tool_calls=None):
    return {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": content,
                    "reasoning_content": reasoning,
                    "tool_calls": tool_calls,
                },
                "finish_reason": finish,
            }
        ]
    }


def _chat_usage(tokens, prompt_tokens=13):
    usage = {"completion_tokens": tokens}
    if prompt_tokens is not None:
        usage["prompt_tokens"] = prompt_tokens
    return {"choices": [], "usage": usage}


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
        backend = self.args.backend
        return asyncio.run(
            serving.benchmark(
                backend=backend,
                api_url=base_url + serving._BACKEND_API_PATHS[backend],
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

    def _run_chat(self, reply, *, extra=None):
        self.args.backend = "sglang-oai-chat"
        with _native_server(reply) as (server, url):
            request = self._request(url)
            request.api_url = url + "/v1/chat/completions"
            request.extra_request_body = extra if extra is not None else {}
            result = asyncio.run(serving.async_request_openai_chat_completions(request))
            self.assertGreater(result.end_time, result.start_time)
            self.assertAlmostEqual(result.latency, result.end_time - result.start_time)
            return result, server.bodies

    def test_chat_real_usage_overrides_output_budget_and_prompt_estimate(self):
        extra = {"stream_options": {"continuous_usage_stats": True}}
        usage = _chat_usage(2)
        usage["usage"]["prompt_tokens_details"] = {"cached_tokens": 7}
        result, bodies = self._run_chat(
            {
                "chunks": [
                    _chat_delta(reasoning="thinking "),
                    _chat_delta("answer", finish="length"),
                    usage,
                ]
            },
            extra=extra,
        )
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.generated_text, "thinking answer")
        self.assertEqual((result.output_len, result.completion_tokens), (2, 2))
        self.assertEqual((result.prompt_len, result.server_prompt_tokens), (13, 13))
        self.assertEqual(result.cached_tokens, 7)
        self.assertEqual(result.finish_reason, "length")
        self.assertIsNone(result.spec_verify_ct)
        self.assertEqual(extra, {"stream_options": {"continuous_usage_stats": True}})
        self.assertEqual(
            bodies[0]["stream_options"],
            {"include_usage": True, "continuous_usage_stats": True},
        )
        self.assertNotIn("return_meta_info", bodies[0])
        self.assertGreater(result.ttft, 0)
        metrics, _ = serving.calculate_metrics(
            [DatasetRow(prompt="estimate", prompt_len=8, output_len=16)],
            [result],
            1,
            _Tokenizer(),
            "sglang-oai-chat",
        )
        self.assertEqual(metrics.total_input, 13)
        self.assertEqual(metrics.total_input_text, 13)
        self.assertEqual(metrics.total_output, 2)

    def test_chat_terminal_reason_or_done_is_sufficient_with_final_usage(self):
        for reason in (
            "stop",
            "length",
            "tool_calls",
            "function_call",
            "content_filter",
            None,
        ):
            with self.subTest(reason=reason):
                result, _ = self._run_chat(
                    {
                        "chunks": [
                            _chat_delta("answer", finish=reason),
                            _chat_usage(1),
                        ],
                        "done": reason is None,
                    }
                )
                self.assertTrue(result.success, result.error)
                self.assertEqual(result.finish_reason, reason)

    def test_chat_partial_eof_and_error_never_count_successful_tokens(self):
        replies = (
            {"chunks": [_chat_delta("partial"), _chat_usage(2)], "done": False},
            {
                "chunks": [
                    _chat_delta("partial"),
                    _chat_usage(2),
                    {"error": {"message": "upstream disconnected"}},
                ]
            },
            {
                "chunks": [
                    _chat_delta("partial"),
                    _chat_usage(2),
                    _chat_delta(finish="abort"),
                ]
            },
            {
                "chunks": [
                    _chat_delta("partial"),
                    _chat_usage(2),
                    _chat_delta(finish="error"),
                ]
            },
        )
        for reply in replies:
            with self.subTest(reply=reply):
                result, _ = self._run_chat(reply)
                self.assertFalse(result.success)
                self.assertEqual(result.http_status, 200)
                self.assertEqual(result.completion_tokens, 2)
                self.assertEqual(result.generated_text, "partial")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    metrics, lengths = serving.calculate_metrics(
                        None, [result], 1, _Tokenizer(), "sglang-oai-chat"
                    )
                self.assertEqual(
                    (metrics.completed, metrics.total_output, lengths), (0, 0, [0])
                )

    def test_chat_missing_usage_and_disabled_usage_are_auditable_failures(self):
        replies = (
            {"chunks": []},
            {"chunks": [_chat_delta("answer", finish="length")]},
            {"chunks": [{"error": {"message": "failed"}}]},
            {"chunks": [_chat_usage(1)]},
        )
        for reply in replies:
            with self.subTest(reply=reply):
                result, _ = self._run_chat(reply)
                self.assertFalse(result.success)
                self.assertTrue(result.error)
        options = {"stream_options": {"include_usage": False}}
        result, bodies = self._run_chat(
            {"chunks": [_chat_delta("answer", finish="stop")]},
            extra=options,
        )
        self.assertFalse(result.success)
        self.assertIsNone(result.completion_tokens)
        self.assertIn("stream_options.include_usage must not be disabled", result.error)
        self.assertIs(bodies[0]["stream_options"]["include_usage"], False)

    def test_chat_intermediate_or_invalid_usage_does_not_finalize_accounting(self):
        intermediate = _chat_delta("partial")
        intermediate["usage"] = {"completion_tokens": 1}
        replies = [
            {"chunks": [intermediate, _chat_delta(finish="length")]},
            {"chunks": [_chat_delta("partial"), _chat_usage(2), _chat_usage(1)]},
        ]
        replies.extend(
            {"chunks": [_chat_delta("answer", finish="stop"), _chat_usage(value)]}
            for value in (True, -1, "2", None)
        )
        for reply in replies:
            with self.subTest(reply=reply):
                result, _ = self._run_chat(reply)
                self.assertFalse(result.success)
                self.assertTrue(result.error)

    def test_chat_tool_only_and_zero_output_have_honest_timing(self):
        tool = [{"index": 0, "function": {"name": "lookup", "arguments": "{}"}}]
        result, _ = self._run_chat(
            {
                "chunks": [
                    _chat_delta(tool_calls=tool),
                    _chat_delta(finish="tool_calls"),
                    _chat_usage(4),
                ]
            }
        )
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.generated_text, "")
        self.assertGreater(result.ttft, 0)
        self.assertEqual(result.itl, [])
        metrics, _ = serving.calculate_metrics(
            None, [result], 1, _Tokenizer(), "sglang-oai-chat", accept_length=9
        )
        self.assertEqual(metrics.total_output, 4)
        zero, _ = self._run_chat(
            {"chunks": [_chat_delta(finish="stop"), _chat_usage(0)]}
        )
        self.assertTrue(zero.success, zero.error)
        self.assertEqual(zero.ttft, 0)
        metrics, _ = serving.calculate_metrics(
            None, [zero], 1, _Tokenizer(), "sglang-oai-chat"
        )
        self.assertIsNone(metrics.mean_ttft_ms)
        self.assertIsNone(metrics.max_output_tokens_per_s)

    def test_chat_nonstream_usage_and_actual_optional_spec_metadata(self):
        self.args.disable_stream = True
        for location in ("sglext", "meta_info", "absent"):
            with self.subTest(location=location):
                body = {
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "reasoning_content": "think ",
                                "content": "answer",
                            },
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {"completion_tokens": 6, "prompt_tokens": 13},
                }
                counters = {
                    "spec_verify_ct": 2,
                    "spec_num_correct_drafts": 3,
                    "spec_num_proposed_drafts": 8,
                }
                if location == "sglext":
                    body["sglext"] = {"spec_tokens_details": counters}
                elif location == "meta_info":
                    body["choices"][0]["meta_info"] = counters
                result, bodies = self._run_chat({"json": body})
                self.assertTrue(result.success, result.error)
                self.assertFalse(bodies[0]["stream"])
                self.assertEqual(result.generated_text, "think answer")
                self.assertEqual(result.completion_tokens, 6)
                metrics = serving._request_speculative_metrics([result])
                self.assertEqual(
                    metrics["accept_length"], None if location == "absent" else 3
                )
                self.assertEqual(
                    metrics["accept_rate"], None if location == "absent" else 3 / 8
                )
        result, _ = self._run_chat(
            {
                "json": {
                    "choices": [
                        {"message": {"content": "answer"}, "finish_reason": "stop"}
                    ]
                }
            }
        )
        self.assertFalse(result.success)
        self.assertIsNone(result.completion_tokens)

    def test_chat_new_server_spec_extension_is_read_from_final_empty_choice_chunk(self):
        result, _ = self._run_chat(
            {
                "chunks": [
                    _chat_delta("answer", finish="length"),
                    {
                        "choices": [],
                        "sglext": {
                            "spec_tokens_details": {
                                "spec_verify_ct": 2,
                                "spec_num_correct_drafts": 3,
                                "spec_num_proposed_drafts": 8,
                            }
                        },
                    },
                    _chat_usage(6),
                ]
            }
        )
        self.assertTrue(result.success, result.error)
        self.assertEqual(
            serving._request_speculative_metrics([result])["accept_length"], 3
        )

    def test_chat_missing_prompt_usage_retains_explicit_estimate_source(self):
        missing, _ = self._run_chat(
            {"chunks": [_chat_delta("answer"), _chat_usage(1, None)]}
        )
        present, _ = self._run_chat(
            {"chunks": [_chat_delta("answer"), _chat_usage(1, 0)]}
        )
        self.assertTrue(missing.success, missing.error)
        self.assertTrue(present.success, present.error)
        self.assertIsNone(missing.server_prompt_tokens)
        self.assertEqual(present.server_prompt_tokens, 0)
        self.assertEqual(missing.prompt_len, 8)
        accounting = serving._input_token_accounting(
            [missing, present], "sglang-oai-chat"
        )
        self.assertEqual(
            accounting["total_input_tokens_source"], "mixed_server_usage_and_dataset"
        )
        self.assertEqual(accounting["requests_with_server_prompt_tokens"], 1)
        self.assertIsNone(accounting["server_prompt_tokens_total"])
        metrics, _ = serving.calculate_metrics(
            None, [missing, present], 1, _Tokenizer(), "sglang-oai-chat"
        )
        self.assertEqual(metrics.total_input, 8)

    def test_chat_all_failed_run_keeps_http_and_partial_audit_details(self):
        self.args.backend = "sglang-oai-chat"
        with (
            _native_server(
                {"chunks": [_chat_delta("partial"), _chat_usage(2)], "done": False}
            ) as (_, url),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            result = self._benchmark(url)
        saved = self._saved_result()
        self.assertEqual(
            (
                saved["completed"],
                saved["failed_requests"],
                saved["total_output_tokens"],
            ),
            (0, 2, 0),
        )
        self.assertEqual(saved["http_status_counts"], {"200": 2})
        self.assertIsNone(saved["mean_tpot_ms"])
        self.assertEqual(result["request_details"], saved["request_details"])
        for detail in saved["request_details"]:
            self.assertEqual(detail["completion_tokens"], 2)
            self.assertEqual(detail["server_prompt_tokens"], 13)
            self.assertEqual(detail["prompt_tokens_source"], "server")

    def test_chat_success_summary_uses_server_prompt_counts_with_coverage(self):
        self.args.backend = "sglang-oai-chat"
        with _native_server(
            {"chunks": [_chat_delta("answer", finish="length"), _chat_usage(3)]}
        ) as (_, url):
            self._benchmark(url)
        saved = self._saved_result()
        self.assertEqual(saved["total_input_tokens"], 26)
        self.assertEqual(saved["total_output_tokens"], 6)
        self.assertIsNone(saved["max_output_tokens_per_s"])
        self.assertEqual(
            saved["max_output_tokens_per_s_source"],
            "unavailable_without_per_token_timestamps",
        )
        self.assertEqual(
            saved["input_token_accounting"]["total_input_tokens_source"], "server_usage"
        )
        self.assertEqual(
            saved["input_token_accounting"]["server_prompt_tokens_total"], 26
        )
        self.assertEqual(
            saved["request_speculative_metrics"]["requests_with_verify_counters"], 0
        )
        self.assertIsNone(saved["request_speculative_metrics"]["accept_length"])

    def test_other_chat_backend_retains_legacy_success_and_payload(self):
        self.args.backend = "vllm-chat"
        with _native_server({"chunks": [_chat_delta("answer")]}) as (server, url):
            request = self._request(url)
            request.api_url = url + "/v1/chat/completions"
            result = asyncio.run(serving.async_request_openai_chat_completions(request))
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output_len, 16)
        self.assertNotIn("stream_options", server.bodies[0])

    def test_chat_plot_does_not_label_chunk_counts_as_tokens(self):
        output = serving.RequestFuncOutput(
            success=True,
            start_time=1,
            end_time=2,
            latency=1,
            output_len=1024,
            completion_tokens=1024,
            ttft=0.1,
            itl=[0.3],
        )
        plot = MagicMock()
        with (
            patch.dict(
                "sys.modules", {"termplotlib": SimpleNamespace(figure=lambda: plot)}
            ),
            patch.object(serving, "TERM_PLOTLIB_AVAILABLE", True),
        ):
            metrics, _ = serving.calculate_metrics(
                None,
                [output],
                1,
                _Tokenizer(),
                "sglang-oai-chat",
                plot_throughput=True,
            )
        self.assertIsNone(metrics.max_output_tokens_per_s)
        self.assertIn("not tokens", plot.plot.call_args_list[0].kwargs["title"])

    def test_chat_http_rejections_and_unsupported_multiple_choices_are_failures(self):
        for status in (429, 503):
            with self.subTest(status=status):
                result, _ = self._run_chat(
                    {"status": status, "json": {"error": "capacity unavailable"}}
                )
                self.assertFalse(result.success)
                self.assertEqual(result.http_status, status)
                self.assertIsNone(result.completion_tokens)
                self.assertIn("capacity unavailable", result.error)
        result, bodies = self._run_chat({"chunks": []}, extra={"n": 2})
        self.assertFalse(result.success)
        self.assertIsNone(result.http_status)
        self.assertEqual(bodies, [])
        self.assertIn("requires n=1", result.error)


if __name__ == "__main__":
    unittest.main()
