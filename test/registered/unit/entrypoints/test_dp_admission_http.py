"""Admission rejects must reach HTTP PD routing before streaming headers."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from starlette.background import BackgroundTask

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints import http_server  # noqa: E402
from sglang.srt.entrypoints.openai.protocol import (  # noqa: E402
    ChatCompletionRequest,
    CompletionRequest,
)
from sglang.srt.entrypoints.openai.serving_chat import (  # noqa: E402
    OpenAIServingChat,
)
from sglang.srt.entrypoints.openai.serving_completions import (  # noqa: E402
    OpenAIServingCompletion,
)
from sglang.srt.managers.io_struct import GenerateReqInput  # noqa: E402
from sglang.srt.managers.tokenizer_manager import (  # noqa: E402
    ReqState,
    TokenizerManager,
)
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats  # noqa: E402

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _Tokenizer:
    """Replace GPU work, while using the production abort/status handling."""

    def __init__(
        self, status=None, *, admission=True, wait_before=None, wait_after=None
    ):
        self.status = status
        self.admission = admission
        self.wait_before = wait_before
        self.wait_after = wait_after
        self.started = False
        self.closed = False
        self.aborted = 0
        self.request_logger = SimpleNamespace(log_requests=False)
        self.manager = TokenizerManager.__new__(TokenizerManager)
        self.manager.rid_to_state = {}
        self.manager.enable_lora = False

    async def generate_request(self, obj, request=None):
        self.started = True
        try:
            if self.wait_before is not None:
                await self.wait_before.wait()
            if self.status is not None:
                state = ReqState(
                    [], True, asyncio.Event(), obj, APIServerReqTimeStats()
                )
                self.manager.rid_to_state[obj.rid] = state
                reason = {
                    "type": "abort",
                    "status_code": self.status,
                    "message": "DP capacity unavailable",
                }
                if self.admission:
                    reason["dp_admission_rejected"] = 1
                out = {"meta_info": {"finish_reason": reason}}
                result = await self.manager._handle_abort_finish_reason(
                    out, state, is_stream=obj.stream
                )
                if result is not None:
                    yield result
                return
            yield {"text": "ok"}
            if self.wait_after is not None:
                await self.wait_after.wait()
        finally:
            self.closed = True

    def create_abort_task(self, obj):
        def abort():
            self.aborted += 1

        return BackgroundTask(abort)


class TestDPAdmissionHTTP(unittest.TestCase):
    def setUp(self):
        self.prior_state = http_server.get_global_state()
        self.addCleanup(http_server.set_global_state, self.prior_state)

    def _native_app(self, tokenizer):
        http_server.set_global_state(SimpleNamespace(tokenizer_manager=tokenizer))
        app = FastAPI()
        app.add_api_route("/generate", http_server.generate_request, methods=["POST"])
        app.add_exception_handler(
            HTTPException, http_server.app.exception_handlers[HTTPException]
        )
        return app

    def test_native_admission_status_is_preserved_for_stream_and_non_stream(self):
        for stream in (False, True):
            for status in (400, 429, 503):
                with self.subTest(stream=stream, status=status):
                    tokenizer = _Tokenizer(status)
                    with patch(
                        "sglang.srt.entrypoints.http_server.get_parallel",
                        return_value=SimpleNamespace(enable_dp_admission=True),
                    ):
                        with TestClient(self._native_app(tokenizer)) as client:
                            response = client.post(
                                "/generate", json={"text": "hello", "stream": stream}
                            )
                    self.assertEqual(response.status_code, status)
                    self.assertEqual(response.json()["code"], status)
                    self.assertNotIn(
                        "text/event-stream", response.headers["content-type"]
                    )
                    self.assertEqual(tokenizer.manager.rid_to_state, {})
                    self.assertTrue(tokenizer.closed)

    def test_default_stream_abort_remains_an_sse_response(self):
        tokenizer = _Tokenizer(503, admission=False)
        with patch(
            "sglang.srt.entrypoints.http_server.get_parallel",
            return_value=SimpleNamespace(enable_dp_admission=False),
        ):
            with TestClient(self._native_app(tokenizer)) as client:
                response = client.post(
                    "/generate", json={"text": "hello", "stream": True}
                )
        self.assertEqual(response.status_code, 200)
        self.assertIn('"status_code":503', response.text)
        self.assertTrue(response.text.endswith("data: [DONE]\n\n"))

    def test_success_stream_is_unchanged_by_admission_prefetch(self):
        bodies = []
        for enabled in (False, True):
            tokenizer = _Tokenizer()
            with patch(
                "sglang.srt.entrypoints.http_server.get_parallel",
                return_value=SimpleNamespace(enable_dp_admission=enabled),
            ):
                with TestClient(self._native_app(tokenizer)) as client:
                    response = client.post(
                        "/generate", json={"text": "hello", "stream": True}
                    )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(tokenizer.closed)
            self.assertEqual(tokenizer.aborted, 1)
            bodies.append(response.content)
        self.assertEqual(bodies, [b'data: {"text":"ok"}\n\ndata: [DONE]\n\n'] * 2)

    def test_openai_streaming_handlers_preserve_admission_status(self):
        # Exercise the actual Chat/Completions generators, their first-chunk
        # prefetch, and OpenAIServingBase.handle_request's HTTP error mapping.
        for serving_class, request_class, fields, module in (
            (
                OpenAIServingChat,
                ChatCompletionRequest,
                {"messages": [{"role": "user", "content": "hello"}]},
                "serving_chat",
            ),
            (
                OpenAIServingCompletion,
                CompletionRequest,
                {"prompt": "hello"},
                "serving_completions",
            ),
        ):
            for status in (400, 429, 503):
                with self.subTest(handler=module, status=status):
                    tokenizer = _Tokenizer(status)
                    serving = serving_class.__new__(serving_class)
                    serving.tokenizer_manager = tokenizer
                    serving._validate_request = Mock(return_value=None)
                    serving._convert_to_internal_request = Mock(
                        side_effect=lambda request, raw_request: (
                            GenerateReqInput(text="hello", stream=True),
                            request,
                        )
                    )
                    app = FastAPI()

                    async def endpoint(raw_request: Request):
                        request = request_class(**(await raw_request.json()))
                        return await serving.handle_request(request, raw_request)

                    app.add_api_route("/test", endpoint, methods=["POST"])
                    with patch(
                        f"sglang.srt.entrypoints.openai.{module}.get_serving",
                        return_value=SimpleNamespace(
                            stream_response_default_include_usage=False,
                            return_input_ids=False,
                            return_output_ids=False,
                        ),
                    ):
                        with TestClient(app) as client:
                            response = client.post(
                                "/test",
                                json={"model": "test", "stream": True, **fields},
                            )
                    self.assertEqual(response.status_code, status, response.text)
                    self.assertEqual(response.json()["code"], status)
                    self.assertTrue(tokenizer.closed)

    def test_unsupported_routes_reject_before_generation_only_when_enabled(self):
        for path, handler, state_name, method, fields in (
            (
                "/v1/responses",
                http_server.v1_responses_request,
                "openai_serving_responses",
                "create_responses",
                {"input": "hello"},
            ),
            (
                "/v1/messages",
                http_server.anthropic_v1_messages,
                "anthropic_serving",
                "handle_messages",
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 4,
                },
            ),
        ):
            for enabled in (False, True):
                for stream in (False, True):
                    with self.subTest(path=path, enabled=enabled, stream=stream):
                        delegate = AsyncMock(return_value={"ok": True})
                        app = FastAPI()
                        app.add_api_route(path, handler, methods=["POST"])
                        app.add_exception_handler(
                            HTTPException,
                            http_server.app.exception_handlers[HTTPException],
                        )
                        setattr(
                            app.state, state_name, SimpleNamespace(**{method: delegate})
                        )
                        with patch(
                            "sglang.srt.entrypoints.http_server.get_parallel",
                            return_value=SimpleNamespace(enable_dp_admission=enabled),
                        ):
                            with TestClient(app) as client:
                                response = client.post(
                                    path,
                                    json={"model": "test", "stream": stream, **fields},
                                )
                        rejected = enabled and (path == "/v1/responses" or stream)
                        if rejected:
                            self.assertEqual(response.status_code, 400, response.text)
                            self.assertIn(
                                "DP admission", response.json()["error"]["message"]
                            )
                            delegate.assert_not_called()
                        else:
                            self.assertEqual(response.status_code, 200, response.text)
                            delegate.assert_awaited_once()


class TestDPAdmissionStreamCleanup(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.prior_state = http_server.get_global_state()
        self.addCleanup(http_server.set_global_state, self.prior_state)
        parallel_patch = patch(
            "sglang.srt.entrypoints.http_server.get_parallel",
            return_value=SimpleNamespace(enable_dp_admission=True),
        )
        parallel_patch.start()
        self.addCleanup(parallel_patch.stop)

    async def test_cancel_during_prefetch_closes_source(self):
        tokenizer = _Tokenizer(wait_before=asyncio.Event())
        http_server.set_global_state(SimpleNamespace(tokenizer_manager=tokenizer))
        pending = asyncio.create_task(
            http_server.serve_generate_request(
                GenerateReqInput(text="hello", stream=True), None
            )
        )
        await asyncio.sleep(0)
        self.assertTrue(tokenizer.started)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertTrue(tokenizer.closed)

    async def test_cancel_after_prefetched_chunk_closes_source(self):
        tokenizer = _Tokenizer(wait_after=asyncio.Event())
        http_server.set_global_state(SimpleNamespace(tokenizer_manager=tokenizer))
        response = await http_server.serve_generate_request(
            GenerateReqInput(text="hello", stream=True), None
        )
        self.assertEqual(
            await anext(response.body_iterator), b'data: {"text":"ok"}\n\n'
        )
        pending = asyncio.create_task(anext(response.body_iterator))
        await asyncio.sleep(0)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertTrue(tokenizer.closed)
        await response.body_iterator.aclose()

    async def test_default_mode_still_defers_generation_until_body_iteration(self):
        tokenizer = _Tokenizer()
        http_server.set_global_state(SimpleNamespace(tokenizer_manager=tokenizer))
        with patch(
            "sglang.srt.entrypoints.http_server.get_parallel",
            return_value=SimpleNamespace(enable_dp_admission=False),
        ):
            response = await http_server.serve_generate_request(
                GenerateReqInput(text="hello", stream=True), None
            )
        self.assertFalse(tokenizer.started)
        await anext(response.body_iterator)
        self.assertEqual(await anext(response.body_iterator), b"data: [DONE]\n\n")
        await response.body_iterator.aclose()
        self.assertTrue(tokenizer.closed)


if __name__ == "__main__":
    unittest.main()
