import asyncio
import unittest

from sglang.srt.utils.utf8_bom_middleware import Utf8BomStripMiddleware
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestUtf8BomMiddleware(unittest.TestCase):
    def run_body(self, chunks, content_type=b"application/json"):
        async def run():
            received = []
            messages = iter(
                {"type": "http.request", "body": body, "more_body": i < len(chunks) - 1}
                for i, body in enumerate(chunks)
            )

            async def receive():
                return next(messages)

            async def downstream(scope, receive, send):
                while True:
                    event = await receive()
                    received.append(event)
                    if not event.get("more_body"):
                        break

            await Utf8BomStripMiddleware(downstream)(
                {"type": "http", "headers": [(b"content-type", content_type)]},
                receive,
                None,
            )
            return b"".join(x["body"] for x in received)

        return asyncio.run(run())

    def test_complete_and_split_bom(self):
        for chunks in (
            [b'\xef\xbb\xbf{"x":1}'],
            [b"\xef", b"\xbb", b"\xbf", b'{"x":1}'],
            [b"", b"\xef\xbb", b'\xbf{"x":', b"1}"],
        ):
            with self.subTest(chunks=chunks):
                self.assertEqual(self.run_body(chunks), b'{"x":1}')

    def test_json_media_type_parameters_are_supported(self):
        self.assertEqual(
            self.run_body([b"\xef\xbb\xbf{}"], b"Application/JSON; charset=utf-8"),
            b"{}",
        )

    def test_non_json_and_nonleading_bom_are_preserved(self):
        payload = b'\xef\xbb\xbf{"x":1}'
        self.assertEqual(self.run_body([payload], b"text/plain"), payload)
        self.assertEqual(
            self.run_body([b'{"x":"', b'\xef\xbb\xbf"}']), b'{"x":"\xef\xbb\xbf"}'
        )

    def test_short_incomplete_prefix_is_not_discarded(self):
        for payload in (b"", b"\xef", b"\xef\xbb", b"{}"):
            self.assertEqual(self.run_body([payload]), payload)


if __name__ == "__main__":
    unittest.main()
