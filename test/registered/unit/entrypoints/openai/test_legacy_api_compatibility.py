"""Contracts retained when migrating the deployed patch stack to v0.5.22."""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from test_serving_chat import _MockTemplateManager, _MockTokenizerManager

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, ToolCall
from sglang.srt.entrypoints.openai.serving_chat import (
    OpenAIServingChat,
    parse_tool_call_arguments,
)
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

FUNCTION_TOOL = {
    "type": "function",
    "function": {"name": "weather", "parameters": {"type": "object"}},
}


class TestLegacyApiCompatibility(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.manager = _MockTokenizerManager()
        self.template = _MockTemplateManager()
        self.chat = OpenAIServingChat(self.manager, self.template)
        self.responses = OpenAIServingResponses(
            self.manager, self.template, chat_serving=self.chat
        )

    @staticmethod
    def request(**updates):
        return ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "weather today"}],
            **updates,
        )

    def test_control_characters_preserve_object_validation(self):
        self.assertEqual(
            parse_tool_call_arguments('{"value":"a\nb"}'), {"value": "a\nb"}
        )
        for invalid in ('{"value":', "[]", '"text"', "123"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_tool_call_arguments(invalid)

    def test_builtin_and_wrapped_tools_preserve_function_tools(self):
        for builtin in (
            {"type": "web_search", "web_search": {"enable": True}},
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "parameters": {"type": "web_search"},
                },
            },
        ):
            with self.subTest(builtin=builtin):
                request = self.request(tools=[builtin, FUNCTION_TOOL])
                self.assertTrue(self.chat.has_builtin_web_search_tools(request))
                self.assertEqual(
                    [t.function.name for t in self.chat._effective_tools(request)],
                    ["weather"],
                )

    def test_web_search_dispatches_to_real_chat_and_preserves_stream(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                request = self.request(
                    tools=[{"type": "web_search"}, FUNCTION_TOOL], stream=streaming
                )
                original = request.model_dump()
                self.responses._run_builtin_web_search_query = AsyncMock(
                    return_value="Forecast: sunny"
                )
                sentinel = object()
                self.chat.handle_request = AsyncMock(return_value=sentinel)
                result = asyncio.run(
                    self.responses.handle_chat_completions_compatible_request(
                        request, Mock()
                    )
                )
                self.assertIs(result, sentinel)
                forwarded = self.chat.handle_request.call_args.args[0]
                self.assertEqual(forwarded.stream, streaming)
                self.assertEqual(forwarded.tools[0].function.name, "weather")
                self.assertIn("Forecast: sunny", forwarded.messages[0].content)
                self.assertEqual(forwarded.messages[-1].role, "user")
                self.assertEqual(request.model_dump(), original)
                self.assertEqual(self.chat._request_id_prefix(), "chatcmpl-")

    def test_native_kimi_template_receives_only_function_tools(self):
        # Responses may be disabled when its optional Harmony vocabulary is
        # unavailable. The ordinary Chat fallback must still filter builtins.
        self.chat.chat_encoding_spec = "kimi_k3"
        self.manager.tokenizer.apply_chat_template = Mock(return_value=[1, 2])
        request = self.request(tools=[{"type": "web_search"}, FUNCTION_TOOL])
        result = self.chat._encode_messages(
            [message.model_dump() for message in request.messages], request, None
        )
        self.assertEqual(result, [1, 2])
        tools = self.manager.tokenizer.apply_chat_template.call_args.kwargs["tools"]
        self.assertEqual([tool["function"]["name"] for tool in tools], ["weather"])

    def test_named_builtin_choice_is_consumed_before_function_parsing(self):
        request = self.request(
            tools=[{"type": "web_search"}, FUNCTION_TOOL],
            tool_choice={"type": "function", "function": {"name": "web_search"}},
        )
        forwarded = self.responses._without_builtin_web_search(request)
        self.assertEqual(forwarded.tool_choice, "auto")
        self.assertEqual([t.function.name for t in forwarded.tools], ["weather"])

    def test_none_or_disabled_builtin_never_issues_search(self):
        for updates in (
            {"tool_choice": "none", "tools": [{"type": "web_search"}]},
            {"tools": [{"type": "web_search", "web_search": {"enable": "false"}}]},
        ):
            with self.subTest(updates=updates):
                self.responses._run_builtin_web_search_query = AsyncMock()
                self.chat.handle_request = AsyncMock(return_value="chat")
                result = asyncio.run(
                    self.responses.handle_chat_completions_compatible_request(
                        self.request(**updates), Mock()
                    )
                )
                self.assertEqual(result, "chat")
                self.responses._run_builtin_web_search_query.assert_not_awaited()
                self.assertIsNone(self.chat.handle_request.call_args.args[0].tools)

    def test_required_search_failure_uses_chat_error_contract(self):
        self.responses._run_builtin_web_search_query = AsyncMock(
            side_effect=RuntimeError("offline")
        )
        self.chat.handle_request = AsyncMock()
        response = asyncio.run(
            self.responses.handle_chat_completions_compatible_request(
                self.request(tools=[{"type": "web_search"}], tool_choice="required"),
                Mock(),
            )
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(json.loads(response.body)["type"], "BadRequestError")
        self.chat.handle_request.assert_not_awaited()

    def test_auto_search_failure_keeps_native_chat_fallback(self):
        self.responses._run_builtin_web_search_query = AsyncMock(
            side_effect=RuntimeError("offline")
        )
        self.chat.handle_request = AsyncMock(return_value="chat")
        result = asyncio.run(
            self.responses.handle_chat_completions_compatible_request(
                self.request(tools=[{"type": "web_search"}]), Mock()
            )
        )
        self.assertEqual(result, "chat")
        self.assertEqual(self.chat.handle_request.call_args.args[0].tool_choice, "none")

    def test_context_errors_return_413_without_reclassifying_other_errors(self):
        for message in (
            "input exceeds the model's maximum context length",
            "input is longer than the model's context length",
        ):
            response = self.chat.create_error_response(message)
            self.assertEqual(response.status_code, 413)
            self.assertEqual(json.loads(response.body)["code"], 413)
            streaming = json.loads(self.chat.create_streaming_error_response(message))
            self.assertEqual(streaming["error"]["code"], 413)
        self.assertEqual(
            self.chat.create_error_response("queue full", status_code=429).status_code,
            429,
        )

    def test_tool_metadata_only_on_first_delta_and_created_is_stable(self):
        parser = Mock()
        parser.parse_stream_chunk.return_value = (
            "",
            [
                SimpleNamespace(name="weather", tool_index=0, parameters='{"city":"'),
                SimpleNamespace(name=None, tool_index=0, parameters='杭州"}'),
            ],
        )
        content = {
            "meta_info": {
                "id": "chatcmpl-tools",
                "prompt_tokens": 7,
                "completion_tokens": 2,
            }
        }
        self.chat._process_tool_call_id = lambda *_: "call-weather"

        async def collect():
            return [
                json.loads(chunk[6:])
                async for chunk in self.chat._process_tool_call_stream(
                    0,
                    "",
                    {0: parser},
                    content,
                    self.request(tools=[FUNCTION_TOOL]),
                    {},
                    123,
                    continuous_usage_stats=True,
                )
            ]

        chunks = asyncio.run(collect())
        first, tail = [c["choices"][0]["delta"]["tool_calls"][0] for c in chunks]
        self.assertEqual(first["id"], "call-weather")
        self.assertEqual(first["type"], "function")
        self.assertEqual(first["function"]["name"], "weather")
        self.assertNotIn("id", tail)
        self.assertNotIn("type", tail)
        self.assertNotIn("name", tail["function"])
        self.assertEqual(
            json.loads(first["function"]["arguments"] + tail["function"]["arguments"]),
            {"city": "杭州"},
        )
        for chunk in chunks:
            self.assertEqual(chunk["created"], 123)
            self.assertEqual(chunk["object"], "chat.completion.chunk")
            self.assertEqual(chunk["usage"]["completion_tokens"], 2)
        self.assertEqual(
            ToolCall(function={"name": "weather", "arguments": "{}"}).type, "function"
        )


if __name__ == "__main__":
    unittest.main()
