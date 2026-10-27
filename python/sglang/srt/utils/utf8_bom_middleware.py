"""ASGI middleware that strips a leading UTF-8 BOM from JSON request bodies."""

from __future__ import annotations

from typing import Any

UTF8_BOM = b"\xef\xbb\xbf"


def _content_type_is_json(scope: dict[str, Any]) -> bool:
    for key, value in scope.get("headers", []):
        if key.lower() != b"content-type":
            continue
        media_type = value.decode("latin-1").split(";", maxsplit=1)[0].strip().lower()
        return media_type == "application/json"
    return False


class Utf8BomStripMiddleware:
    """Strip a leading UTF-8 BOM before JSON body parsing sees the payload."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not _content_type_is_json(scope):
            await self.app(scope, receive, send)
            return

        pending = bytearray()
        checked_prefix = False

        async def receive_without_bom():
            nonlocal checked_prefix

            while True:
                message = await receive()
                if message["type"] != "http.request" or checked_prefix:
                    return message

                pending.extend(message.get("body", b""))
                more_body = message.get("more_body", False)

                if more_body and len(pending) < len(UTF8_BOM):
                    continue

                body = bytes(pending).removeprefix(UTF8_BOM)

                pending.clear()
                checked_prefix = True
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": more_body,
                }

        await self.app(scope, receive_without_bom, send)
