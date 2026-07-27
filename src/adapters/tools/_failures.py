"""One place where an adapter's failures become tool results.

Five connectors each had their own `_format_http_error` differing only in the
vendor's name, and thirty-odd handlers each ended in the same six lines:

    except httpx.HTTPStatusError as e:
        return ToolResult.error(_format_http_error(e))
    except Exception as e:
        return ToolResult.error(f"error: {e}")

That is not error handling a reader learns anything from the thirtieth time,
and a handler that forgot the second clause would raise into the agent loop
instead of answering the model. `api_errors` makes the contract one decorator:
a handler body says what the tool DOES, and failing is handled the same way
everywhere by construction.

Handlers with a genuinely different failure story — a JSON body to explain, a
404 that means "not found" rather than "broken" — keep their own try blocks.
"""
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import httpx

from ports import ToolResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from ports import ToolContext

    Handler = Callable[[dict[str, Any], ToolContext], Awaitable[ToolResult]]

# Enough of the response body to name the cause, not so much that a wall of
# vendor HTML lands in the conversation.
_BODY_CHARS = 300

# The two status codes these connectors actually branch on. Named here rather
# than five times over, because `== 204` in a request wrapper reads as a magic
# number and "no content" is the thing being tested.
HTTP_OK = 200
HTTP_NO_CONTENT = 204


def format_http_error(vendor: str, e: httpx.HTTPStatusError) -> str:
    """Render a failed API call as the one line the model will read."""
    return f"{vendor} API error {e.response.status_code}: {(e.response.text or '')[:_BODY_CHARS]}"


def handler_errors(render: Callable[[Exception], str]) -> Callable[[Handler], Handler]:
    """Turn any failure inside a handler into ToolResult.error, via `render`.

    The HTTP connectors want the status code and body; the IMAP one wants its
    own wording. What they share is that a tool must ANSWER rather than raise:
    an exception escaping here reaches the agent loop, and the model learns
    nothing it can act on.
    """

    def decorator(handler: Handler) -> Handler:
        @functools.wraps(handler)
        async def wrapper(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
            try:
                return await handler(args, ctx)
            except Exception as e:
                return ToolResult.error(render(e))

        return wrapper

    return decorator


def api_errors(vendor: str) -> Callable[[Handler], Handler]:
    """Turn a handler's HTTP and unexpected failures into ToolResult.error.

    Applied BELOW `@tool`, so it wraps the coroutine before the spec is built.
    """

    def render(e: Exception) -> str:
        if isinstance(e, httpx.HTTPStatusError):
            return format_http_error(vendor, e)
        return f"error: {e}"

    return handler_errors(render)
