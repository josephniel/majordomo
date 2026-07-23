"""OpenAI-Chat-Completions-compatible agents (OpenAI + DeepSeek).

Both vendors use the same wire format, so they share an
`ChatCompletionsAgent` base; concrete classes pin the model + base_url +
env-var name.

These agents now support tool calling: connectors hand over vendor-neutral
`ToolSpec` objects, the agent translates them into OpenAI's `tools=[…]`
function format, runs the standard tool-call loop, and dispatches each call
back to `ToolSpec.handler`. Result: the same memory_save / search_emails /
schedule_create tools work on Claude *or* OpenAI *or* DeepSeek.

History comes from `ConversationHistory` (mirrored by CascadingAgent on
every turn). Each `send()` reads recent rows, formats them as a
`messages=[…]` list, and replays — no server-side session.
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import logging
import os
from typing import Any, Optional

from core import Connector, ToolSpec, as_tool_result

from .base import (
    Agent,
    Attachment,
    PersonaLike,
    ContextBuilder,
    Summarizer,
    ToolUseCallback,
    UsageLimitError,
)
from .history import ConversationHistory

log = logging.getLogger(__name__)


# Cap on how many model→tool→model round trips a single user turn can do
# before we bail. Keeps a misbehaving model from spinning forever.
MAX_TOOL_LOOP_ITERATIONS = 12

# ---- tool subsetting (for token-constrained vendors) ----
# Connectors whose tools are ALWAYS sent — the agent's own faculties, cheap
# and relevant to almost any turn.
ALWAYS_ON_CONNECTORS = frozenset({"memory", "schedule"})

# Connector base name -> trigger keywords. A connector's tools are attached
# only when the user's message contains its base name or one of these. Lists
# are deliberately generous — a missed tool is worse than a few extra.
CONNECTOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "gmail": ("email", "e-mail", "mail", "inbox", "unread", "reply", "send",
              "message", "draft", "compose", "attachment"),
    "google_calendar": ("calendar", "event", "meeting", "appointment", "invite",
                         "schedule", "free", "busy", "availab", "reschedule",
                         "tomorrow", "today", "agenda"),
    "clickup": ("task", "todo", "to-do", "ticket", "clickup", "project",
                "assign", "due", "backlog", "sprint", "status"),
    "splitwise": ("split", "splitwise", "expense", "owe", "owed", "paid",
                  "settle", "reimburse", "bill", "share", "cost", "debt"),
    "yahoo": ("stock", "yahoo", "portfolio", "ticker", "market", "share price",
              "equity", "quote", "index"),
    # In-process faculties added later — keyword-routed, NOT always-on:
    # without entries here they'd hit the unknown-connector fallback and
    # ride every TPM-capped turn.
    "code": ("code", "run", "script", "python", "compute", "calculate",
             "csv", "chart", "graph", "convert", "parse", "generate"),
    "files": ("file", "send", "download", "csv", "chart", "artifact",
              "attachment", "report"),
    "documents": ("document", "doc", "pdf", "file", "search", "saved",
                  "read", "attachment", "notes", "paper", "contract"),
    "skills": ("skill", "always", "never", "remember how", "from now on",
               "procedure", "instructions", "teach"),
    "delegate": ("delegate", "summarize all", "audit", "go through",
                 "digest", "triage", "review all", "every"),
}


_USAGE_LIMIT_HINTS = (
    "rate_limit",
    "rate limit",
    "quota",
    "insufficient_quota",
    "429",
    "too many requests",
    "tokens per min",
    "tpm",
    "rpm",
    "service_unavailable",
    "overloaded",
)


def _signals_usage_limit(exc: BaseException) -> bool:
    """Classify a single exception (no chain walking).

    Prefers the OpenAI SDK's typed, status-code-based exceptions so detection
    doesn't depend on provider error wording; falls back to string/classname
    heuristics for non-SDK or wrapped errors.
    """
    try:
        import openai

        if isinstance(
            exc,
            (openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError),
        ):
            return True
        if isinstance(exc, openai.APIStatusError):
            code = getattr(exc, "status_code", None)
            if code in (408, 409, 429) or (isinstance(code, int) and code >= 500):
                return True
    except Exception:
        # openai unavailable for some reason — fall through to heuristics.
        pass

    msg = (str(exc) or "").lower()
    if any(hint in msg for hint in _USAGE_LIMIT_HINTS):
        return True
    cls_name = exc.__class__.__name__.lower()
    return "ratelimit" in cls_name or "overloaded" in cls_name


def _is_usage_limit(exc: BaseException) -> bool:
    """True if `exc` (or any wrapped cause) is a rate/usage/overload limit that
    should trigger failover. Walks the __cause__/__context__ chain so a limit
    error wrapped in a generic exception is still caught."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if _signals_usage_limit(cur):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _spec_to_openai_function(prefixed_name: str, spec: ToolSpec) -> dict[str, Any]:
    """Translate a ToolSpec into the OpenAI tools[] entry format.

    `json_schema()` is the normalized form — full JSON Schema when the
    connector declared one (required/descriptions/enums travel through),
    permissive object schema for legacy {arg: type} maps.
    """
    return {
        "type": "function",
        "function": {
            "name": prefixed_name,
            "description": spec.description,
            "parameters": spec.json_schema(),
        },
    }


def _fit_tool_name(name: str, taken: dict[str, Any]) -> str:
    """OpenAI caps function names at 64 chars. Truncate, but never let two
    long names silently collapse into the same key — disambiguate with a
    short stable hash suffix."""
    if len(name) <= 64 and name not in taken:
        return name
    base = name[:64]
    if base not in taken:
        return base
    import hashlib
    suffix = hashlib.sha1(name.encode()).hexdigest()[:6]
    return f"{name[:57]}_{suffix}"


def _extract_failed_generation(exc: BaseException) -> Optional[str]:
    """Pull Groq's `failed_generation` string out of a tool_use_failed 400.
    Checks the SDK's parsed body first, then falls back to scraping str(exc)."""
    body = getattr(exc, "body", None)
    candidates = []
    if isinstance(body, dict):
        candidates.append(body)
        if isinstance(body.get("error"), dict):
            candidates.append(body["error"])
    for d in candidates:
        fg = d.get("failed_generation")
        if isinstance(fg, str) and fg:
            return fg
    # Last resort: regex it out of the stringified exception.
    m = re.search(r"'failed_generation':\s*'(.*?)'\}", str(exc), re.DOTALL)
    return m.group(1) if m else None


def _parse_llama_tool_calls(text: str) -> list[tuple[str, str]]:
    """Parse Llama's malformed tool syntax `<function=NAME {json}>` (one or
    more) into (name, arguments_json) pairs. Brace-matches so nested JSON is
    captured correctly."""
    out: list[tuple[str, str]] = []
    for m in re.finditer(r"<function=([A-Za-z0-9_\-.]+)", text or ""):
        name = m.group(1)
        start = text.find("{", m.end())
        if start == -1:
            continue
        depth = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            continue
        candidate = text[start:end + 1]
        try:
            json.loads(candidate)  # only accept valid JSON args
        except json.JSONDecodeError:
            continue
        out.append((name, candidate))
    return out


def _recover_failed_tool_calls(exc: BaseException) -> list[tuple[str, str]]:
    """If `exc` is a Groq/Llama tool_use_failed 400, extract the intended tool
    calls from its failed generation. Returns [] when not applicable, so the
    normal error handling proceeds."""
    marker = "tool_use_failed"
    code = getattr(exc, "code", None)
    if code != marker and marker not in str(exc):
        return []
    fg = _extract_failed_generation(exc)
    if not fg:
        return []
    return _parse_llama_tool_calls(fg)


def _extract_text_from_tool_result(result: Any) -> str:
    """Flatten a handler result (ToolResult, or a legacy MCP-shaped dict from
    external MCP servers) into the single string OpenAI's `tool` message
    content field wants. (The error flag isn't surfaced separately here —
    handlers word their error text self-descriptively.)"""
    return as_tool_result(result).text or "(empty)"


class ChatCompletionsAgent(Agent):
    """Base implementation for any vendor speaking the OpenAI Chat
    Completions API (OpenAI itself, DeepSeek, etc.)."""

    DEFAULT_MODEL: str = ""
    DEFAULT_BASE_URL: Optional[str] = None
    API_KEY_ENV: str = ""
    # Extra params merged into every chat.completions.create() call. Subclasses
    # use this for vendor-specific knobs (e.g. Gemini's reasoning_effort).
    EXTRA_COMPLETION_KWARGS: dict[str, Any] = {}
    # Whether this backend can accept image inputs (OpenAI multimodal parts).
    # Gemini + OpenAI: yes; DeepSeek's chat model: no.
    SUPPORTS_VISION: bool = False

    # Context assembly budget. Rows are included newest-first until the char
    # budget is spent; summary rows ALWAYS ride along (they're small and they
    # are the only memory of everything already compacted). This replaces the
    # old fixed message count, which silently dropped history in
    # short-message chats long before compaction fired (gap A4).
    MAX_HISTORY_CHARS = 24_000  # ≈ 6k tokens
    MAX_HISTORY_FETCH = 200

    # Token-constrained vendors (small TPM) set this True to send only the
    # relevant tools per turn (see _select_tools) instead of the full set.
    SUBSET_TOOLS: bool = False

    def __init__(
        self,
        context_builder: ContextBuilder,
        history: ConversationHistory,
        persona_id: str,
        chat_id: int,
        connectors: Optional[list[Connector]] = None,
        persona: Optional[PersonaLike] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        external_tools_provider: Optional[
            Any  # async () -> dict[str, ToolSpec]; see agents/external_mcp.py
        ] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        self._composer = context_builder
        self._history = history
        self._persona_id = persona_id
        self._chat_id = chat_id
        self._connectors = connectors or []
        self._persona = persona
        self._model = model or self.DEFAULT_MODEL
        self._api_key = api_key or os.environ.get(self.API_KEY_ENV)
        self._base_url = base_url or self.DEFAULT_BASE_URL
        self._client = None
        self._current_task: Optional[asyncio.Task] = None
        self._external_tools_provider = external_tools_provider
        self._external_tools_loaded = False
        self._max_tokens = max_tokens or None  # 0/None → vendor default
        self.last_turn_usage: dict[str, Any] = {}
        if not self._api_key:
            raise RuntimeError(
                f"{self.__class__.__name__}: env var {self.API_KEY_ENV!r} is not set"
            )
        # Resolve once — connectors / persona don't change at runtime.
        # (External MCP tools are merged lazily at first send — they need a
        # running event loop to spawn their subprocesses.)
        self._tools_by_name: dict[str, ToolSpec] = self._collect_tools()
        self._openai_tools: Optional[list[dict[str, Any]]] = None
        self._rebuild_openai_tools()

    def _rebuild_openai_tools(self) -> None:
        self._openai_tools = (
            [_spec_to_openai_function(n, s) for n, s in self._tools_by_name.items()]
            or None
        )
        # Per-tool cache + connector-base map, used by tool subsetting.
        self._openai_tool_by_name = {
            t["function"]["name"]: t for t in (self._openai_tools or [])
        }
        self._tool_connector = {
            name: self._connector_base_for(name) for name in self._tools_by_name
        }

    def _connector_base_for(self, prefixed_name: str) -> str:
        """Map a `<server>__<tool>` key to its connector's base name
        (gmail, google_calendar, splitwise, memory, schedule, …)."""
        server = prefixed_name.rsplit("__", 1)[0]
        for c in self._connectors:
            if c.owns_profile(server):
                return c.name
        return server

    def _select_tools(self, text: str) -> Optional[list[dict[str, Any]]]:
        """Choose which tools to send this turn.

        Token-constrained vendors (SUBSET_TOOLS) can't afford all ~60 tool
        schemas every request (Groq free tier is 12k TPM; the full set is
        ~8.8k before history). So we always include the ALWAYS_ON connectors
        (memory, schedule) and attach a connector's tools only when the
        message looks relevant to it — keyword routing with a safe fallback.
        Non-constrained vendors send everything.
        """
        if not self.SUBSET_TOOLS or not self._openai_tools:
            return self._openai_tools
        low = (text or "").lower()
        selected: set[str] = set(ALWAYS_ON_CONNECTORS)
        for base, keywords in CONNECTOR_KEYWORDS.items():
            if base in low or any(k in low for k in keywords):
                selected.add(base)
        tools = [
            t for name, t in self._openai_tool_by_name.items()
            if self._tool_connector.get(name) in selected
            or self._tool_connector.get(name) not in CONNECTOR_KEYWORDS  # unknown → always keep
        ]
        return tools or self._openai_tools

    @property
    def session_id(self) -> Optional[str]:
        return None  # client-side history only

    @property
    def model_name(self) -> str:
        return self._model

    async def probe_tool_calling(self) -> tuple[bool, str]:
        """Layer-4 canary: does this model actually invoke a tool when asked?
        Sends a tiny one-tool request (≈100 tokens, safe under any TPM cap).
        Returns (ok, detail). This is what would have caught gemini-flash
        silently regressing to hallucinated saves."""
        try:
            await self.start()
            ping = {
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "Acknowledge readiness. Call this to reply.",
                    "parameters": {"type": "object", "properties": {
                        "ok": {"type": "boolean"}}},
                },
            }
            try:
                resp = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content":
                               "Call the ping tool to confirm you can use tools."}],
                    tools=[ping], tool_choice="auto", max_tokens=128,
                    **self.EXTRA_COMPLETION_KWARGS,
                )
            except Exception as e:
                # A malformed-but-recoverable tool call (Groq/Llama
                # tool_use_failed) means the model DID try to call the tool —
                # and the live tool loop recovers it — so count it as a pass.
                if _recover_failed_tool_calls(e):
                    return (True, "called ping (recovered malformed format)")
                raise
            msg = resp.choices[0].message if resp.choices else None
            called = bool(getattr(msg, "tool_calls", None))
            return (called, "called ping" if called else "no tool_call returned (hallucination risk)")
        except Exception as e:
            return (False, str(e)[:140])

    async def start(self) -> None:
        if self._client is not None:
            return
        from openai import AsyncOpenAI
        kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        # CRITICAL for latency: CascadingAgent IS the retry/failover layer, so
        # the SDK's own retries are redundant AND harmful — on a 429 the SDK
        # would otherwise retry twice with exponential backoff (honoring
        # Retry-After), burning tens of seconds per rate-limited vendor before
        # we ever get to fail over. max_retries=0 makes a busy vendor fail
        # instantly so we advance to the next one immediately. Tight timeout
        # caps a hung request (SDK default is 600s).
        self._client = AsyncOpenAI(max_retries=0, timeout=30.0, **kwargs)

    async def stop(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None

    async def interrupt(self) -> None:
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

    # ---- tool collection ----

    def _collect_tools(self) -> dict[str, ToolSpec]:
        """Walk connectors, apply the persona's allowed_tool_names filter,
        return a flat name→ToolSpec map. Names are prefixed with the server
        so multi-profile connectors (gmail per profile) stay disambiguated.
        """
        out: dict[str, ToolSpec] = {}
        for c in self._connectors:
            allowed: Optional[list[str]] = None
            if self._persona is not None:
                allowed = self._persona.allowed_tool_names(c)
                if allowed == []:
                    continue
            for server_name, specs in c.builtin_servers().items():
                for spec in specs:
                    if allowed is not None and spec.name not in allowed:
                        continue
                    prefixed = _fit_tool_name(f"{server_name}__{spec.name}", out)
                    out[prefixed] = spec
        return out

    # ---- main turn ----

    async def send(
        self,
        text: str,
        on_tool_use: Optional[ToolUseCallback] = None,
        attachments: Optional[list[Attachment]] = None,
        current_row_id: Optional[int] = None,
    ) -> str:
        """`text` IS the current user message — it goes on the wire verbatim
        (including any composed context like the auto-RAG memory block). The
        mirror supplies HISTORY only: when the caller mirrored this turn
        already (CascadingAgent passes the row id as `current_row_id`), that
        raw row is excluded so the message isn't sent twice. This replaced
        the old last-row-must-be-user heuristic, which double-appended on
        mid-turn failover after tool calls and silently DROPPED the memory
        block for every chat-completions vendor."""
        if self._client is None:
            await self.start()
        await self._merge_external_tools()

        history_rows = await self._history.recent(
            self._persona_id, self._chat_id, limit=self.MAX_HISTORY_FETCH,
        )
        if current_row_id is not None:
            history_rows = [r for r in history_rows if r.get("id") != current_row_id]

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._composer.build()},
        ]
        messages.extend(self._assemble_context(history_rows))
        messages.append({"role": "user", "content": text})

        # Multimodal: attach images to the current user turn when this
        # backend can see them.
        if attachments:
            self._apply_attachments(messages, attachments)

        # Route: token-constrained vendors get only the relevant tools.
        tools = self._select_tools(text)

        self._current_task = asyncio.current_task()
        self.last_turn_usage = {}
        try:
            return await self._run_tool_loop(messages, on_tool_use, tools)
        finally:
            self._current_task = None

    async def _merge_external_tools(self) -> None:
        """Fold in tools from external stdio MCP servers (once). Keeps
        connector parity with the Claude path, which mounts these natively —
        without this, a failover silently loses capabilities (gap A3)."""
        if self._external_tools_loaded or self._external_tools_provider is None:
            return
        self._external_tools_loaded = True
        try:
            external: dict[str, ToolSpec] = await self._external_tools_provider()
        except Exception:
            log.exception("external MCP tools unavailable to %s (continuing without)",
                          self.__class__.__name__)
            return
        added = 0
        for name, spec in external.items():
            fitted = _fit_tool_name(name, self._tools_by_name)
            if fitted in self._tools_by_name:
                continue
            self._tools_by_name[fitted] = spec
            added += 1
        if added:
            self._rebuild_openai_tools()
            log.info("%s: merged %d external MCP tools", self.__class__.__name__, added)

    def _assemble_context(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Budget-based replay of the mirror. Newest rows win the budget;
        summary rows always ride along; mirrored tool calls surface as
        inline system notes so this vendor knows what actions were taken
        (possibly by a different vendor)."""
        kept: list[dict[str, Any]] = []
        budget = self.MAX_HISTORY_CHARS
        for row in reversed(rows):
            role = row["role"]
            meta = row.get("metadata") or {}
            if role == "summary":
                kept.append(row)  # always — small, and it's the long-term memory
                continue
            if role == "system" and not meta.get("tool_use"):
                continue  # non-tool system rows aren't replayable
            cost = len(row["content"])
            if budget - cost < 0 and kept:
                continue  # budget spent; only summaries beyond this point
            budget -= cost
            kept.append(row)
        kept.reverse()

        messages: list[dict[str, Any]] = []
        # Summaries first: a summary row's id is NEWER than the raw turns it
        # kept alongside (it's inserted at compaction time), but it covers
        # the OLDEST part of the conversation — rendering it before the raw
        # turns keeps the narrative in causal order.
        for row in kept:
            if row["role"] == "summary":
                messages.append({
                    "role": "system",
                    "content": f"[Earlier conversation summary]\n{row['content']}",
                })
        for row in kept:
            role = row["role"]
            meta = row.get("metadata") or {}
            if role == "summary":
                continue
            if role == "system" and meta.get("tool_use"):
                messages.append({
                    "role": "system",
                    "content": f"[The assistant performed this action: {row['content']}]",
                })
            elif role in ("user", "assistant"):
                messages.append({"role": role, "content": row["content"]})
        return messages

    def _apply_attachments(self, messages: list[dict[str, Any]], attachments: list[Attachment]) -> None:
        """Augment the latest user message with image parts (if this backend
        supports vision) and/or a note about what couldn't be processed."""
        idx = next((i for i in range(len(messages) - 1, -1, -1)
                    if messages[i].get("role") == "user"), None)
        if idx is None:
            return
        base = messages[idx].get("content")
        base_text = base if isinstance(base, str) else ""

        images = [a for a in attachments if (a.media_type or "").startswith("image/")]
        others = [a for a in attachments if not (a.media_type or "").startswith("image/")]

        notes: list[str] = []
        if others:
            notes.append(f"[{len(others)} non-image attachment(s) (e.g. PDF) were sent; "
                         f"they can only be read on the Claude backend.]")
        if images and not self.SUPPORTS_VISION:
            notes.append(f"[{len(images)} image(s) were sent but this model can't view images.]")
        note = ("\n" + " ".join(notes)) if notes else ""

        if images and self.SUPPORTS_VISION:
            parts: list[dict[str, Any]] = [{"type": "text", "text": (base_text + note) or "(image attached)"}]
            for a in images:
                b64 = base64.b64encode(a.data).decode("ascii")
                parts.append({"type": "image_url",
                              "image_url": {"url": f"data:{a.media_type};base64,{b64}"}})
            messages[idx]["content"] = parts
        elif note:
            messages[idx]["content"] = base_text + note

    async def _run_tool_loop(
        self,
        messages: list[dict[str, Any]],
        on_tool_use: Optional[ToolUseCallback],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        if tools is None:
            tools = self._openai_tools
        total_in = 0
        total_out = 0
        for iteration in range(MAX_TOOL_LOOP_ITERATIONS):
            try:
                kwargs: dict[str, Any] = {
                    "model": self._model,
                    "messages": messages,
                    **self.EXTRA_COMPLETION_KWARGS,
                }
                if self._max_tokens:
                    kwargs["max_tokens"] = self._max_tokens
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                resp = await self._client.chat.completions.create(**kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Recover from Groq/Llama's intermittent malformed tool-call
                # syntax: it emits `<function=name {json}>` as text instead of
                # proper tool_calls, and Groq 400s with `tool_use_failed`. We
                # parse the failed generation, run the tools, and continue —
                # so the quirk is transparent and Groq keeps serving.
                recovered = _recover_failed_tool_calls(e)
                if recovered:
                    log.info("%s: recovered %d malformed tool call(s) from a "
                             "tool_use_failed error", self.__class__.__name__, len(recovered))
                    messages.append({
                        "role": "assistant", "content": "",
                        "tool_calls": [
                            {"id": f"recovered_{iteration}_{i}", "type": "function",
                             "function": {"name": n, "arguments": a}}
                            for i, (n, a) in enumerate(recovered)
                        ],
                    })
                    await self._dispatch_calls(
                        [(f"recovered_{iteration}_{i}", n, a)
                         for i, (n, a) in enumerate(recovered)],
                        messages, on_tool_use,
                    )
                    continue
                if _is_usage_limit(e):
                    raise UsageLimitError(
                        f"{self.__class__.__name__} usage limit hit: {e}"
                    ) from e
                raise

            # Accumulate token usage across the whole tool loop.
            usage = getattr(resp, "usage", None)
            if usage is not None:
                total_in += getattr(usage, "prompt_tokens", 0) or 0
                total_out += getattr(usage, "completion_tokens", 0) or 0
                self.last_turn_usage = {
                    "input_tokens": total_in, "output_tokens": total_out,
                }

            choice = resp.choices[0] if resp.choices else None
            msg = choice.message if choice else None
            if msg is None:
                return "(no response)"

            tool_calls = getattr(msg, "tool_calls", None) or []

            # Append the assistant message verbatim to the running stack so the
            # next call sees the tool_calls it asked for. Use model_dump() rather
            # than hand-rebuilding tool_calls: it preserves vendor-specific extras
            # — notably Gemini 3's per-call `extra_content.google.thought_signature`,
            # which Gemini REQUIRES echoed back on the next request or it 400s
            # ("Function call is missing a thought_signature"). Harmless no-op for
            # OpenAI/DeepSeek, which don't emit extra fields.
            asst_entry: dict[str, Any] = msg.model_dump(exclude_none=True)
            asst_entry["role"] = "assistant"
            messages.append(asst_entry)

            if not tool_calls:
                # Final answer.
                return (msg.content or "").strip() or "(no response)"

            await self._dispatch_calls(
                [(tc.id, tc.function.name, tc.function.arguments or "{}")
                 for tc in tool_calls],
                messages, on_tool_use,
            )

        log.warning(
            "%s tool loop exceeded %d iterations; bailing",
            self.__class__.__name__, MAX_TOOL_LOOP_ITERATIONS,
        )
        return (
            "(I ran out of tool-call retries on this turn. "
            "Try simplifying the request or rephrasing.)"
        )

    async def _dispatch_calls(
        self,
        calls: list[tuple[str, str, str]],  # (id, name, arguments_json)
        messages: list[dict[str, Any]],
        on_tool_use: Optional[ToolUseCallback],
    ) -> None:
        """Run each requested tool, appending its `tool` result message.
        Shared by the normal path and the malformed-call recovery path."""
        for call_id, tool_name, arguments in calls:
            try:
                args = json.loads(arguments or "{}")
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {}
            if on_tool_use is not None:
                try:
                    await on_tool_use(tool_name, dict(args))
                except Exception:
                    log.debug("on_tool_use callback raised", exc_info=True)
            spec = self._tools_by_name.get(tool_name)
            if spec is None:
                result_text = f"error: unknown tool {tool_name!r}"
            else:
                try:
                    result = await spec.handler(args)
                    result_text = _extract_text_from_tool_result(result)
                except Exception as e:
                    result_text = f"error: {e}"
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": result_text,
            })


class OpenAIAgent(ChatCompletionsAgent):
    # Paid tier has generous TPM, but the full ~60-tool schema (~7k tokens) is
    # still billed on every turn — subsetting cuts most of it with the same
    # safety nets (always-on connectors, empty→full fallback) as Groq/Gemini.
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_BASE_URL = None
    API_KEY_ENV = "OPENAI_API_KEY"
    REQUIRED_ENV = ["OPENAI_API_KEY"]
    SUPPORTS_VISION = True
    SUBSET_TOOLS = True


class DeepSeekAgent(ChatCompletionsAgent):
    # Subsets tools for the same billed-schema reason as OpenAIAgent.
    DEFAULT_MODEL = "deepseek-chat"
    DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
    API_KEY_ENV = "DEEPSEEK_API_KEY"
    REQUIRED_ENV = ["DEEPSEEK_API_KEY"]
    SUBSET_TOOLS = True


class GeminiAgent(ChatCompletionsAgent):
    # Google Gemini via its OpenAI-compatible endpoint. Override the model with
    # the GEMINI_MODEL env var (see PersonaRuntime); key comes from GEMINI_API_KEY.
    # `gemini-flash-latest` tracks the current flash model. reasoning_effort=low
    # keeps 3.x flash's built-in thinking minimal so the token budget goes to the
    # answer (reasoning_effort="none" is rejected by the API for these models).
    # Pin a specific flash rather than `-latest`: the alias auto-upgrades to
    # the newest generation (e.g. Gemini 3.6), which carries the SMALLEST free
    # quota — a stable prior flash has far more free headroom. Override with
    # GEMINI_MODEL. Also subsets tools (like Groq) to keep per-turn tokens down
    # — without it, Gemini was sending all ~60 tool schemas (~11.7k tok/turn),
    # torching its free quota.
    DEFAULT_MODEL = "gemini-2.5-flash"
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    API_KEY_ENV = "GEMINI_API_KEY"
    REQUIRED_ENV = ["GEMINI_API_KEY"]
    EXTRA_COMPLETION_KWARGS = {"reasoning_effort": "low"}
    SUPPORTS_VISION = True
    SUBSET_TOOLS = True
    MAX_HISTORY_CHARS = 16_000  # ≈ 4k tokens; conserve free-tier quota


class GroqAgent(ChatCompletionsAgent):
    # Groq via its OpenAI-compatible endpoint. Default is Llama 3.3 70B — a
    # strong, reliable function-caller (unlike gemini-flash, which hallucinates
    # tool use), which is why it's the preferred primary for this tool-heavy
    # agent. Override with GROQ_MODEL. Key: GROQ_API_KEY (free at
    # console.groq.com). Llama 3.3 70B is text-only, so image turns fail over
    # to a vision-capable vendor (gemini/claude) further down the chain.
    #
    # Free tier is 12k tokens/minute; the full ~60-tool schema alone is ~6.8k,
    # so SUBSET_TOOLS keeps only relevant tools per turn and the history cap is
    # tightened — together a turn lands well under the limit.
    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
    API_KEY_ENV = "GROQ_API_KEY"
    REQUIRED_ENV = ["GROQ_API_KEY"]
    SUPPORTS_VISION = False
    SUBSET_TOOLS = True
    MAX_HISTORY_CHARS = 10_000  # ≈ 2.5k tokens — leaves headroom under 12k TPM


# Map a PRIMARY_LLM vendor name to its OpenAI-compatible backend class.
_VENDOR_BACKENDS = {
    "groq": GroqAgent,
    "gemini": GeminiAgent,
    "openai": OpenAIAgent,
    "deepseek": DeepSeekAgent,
}


class ChatCompletionsSummarizer(Summarizer):
    """`Summarizer` that runs memory/history compaction through any
    OpenAI-compatible vendor (Gemini/OpenAI/DeepSeek). This is what keeps the
    memory subsystem LLM-agnostic: a Gemini-primary bot summarizes with Gemini,
    not Claude. Reuses the vendor backend's model/base_url/key/extra config so
    there's a single source of truth per vendor.
    """

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None,
                 extra: Optional[dict] = None) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._extra = dict(extra or {})
        self._client = None  # lazy AsyncOpenAI

    @classmethod
    def for_vendor(cls, vendor: str) -> "ChatCompletionsSummarizer":
        backend = _VENDOR_BACKENDS.get(vendor)
        if backend is None:
            raise ValueError(f"no OpenAI-compatible summarizer for vendor {vendor!r}")
        # Per-vendor model override env var, if the operator set one.
        model = os.environ.get({"gemini": "GEMINI_MODEL", "groq": "GROQ_MODEL"}.get(vendor, ""))
        api_key = os.environ.get(backend.API_KEY_ENV)
        if not api_key:
            raise RuntimeError(f"{vendor} summarizer: env var {backend.API_KEY_ENV!r} is not set")
        return cls(
            model=model or backend.DEFAULT_MODEL,
            api_key=api_key,
            base_url=backend.DEFAULT_BASE_URL,
            extra=backend.EXTRA_COMPLETION_KWARGS,
        )

    async def summarize(self, prompt: str, *, deep: bool = False) -> str:
        if self._client is None:
            from openai import AsyncOpenAI
            # Same fast-fail rationale as the agent client (see start()):
            # summarization is best-effort/background, so don't let SDK
            # retries stall it either.
            self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url,
                                       max_retries=0, timeout=30.0)
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            **self._extra,
        )
        return (resp.choices[0].message.content or "").strip()
