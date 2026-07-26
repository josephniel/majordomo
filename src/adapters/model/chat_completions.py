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
from typing import Any, Optional

from ports import ConversationRef, Connector, ToolContext, ToolSpec, as_tool_result

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

# Tool subsetting for token-constrained vendors is driven by what each
# provider DECLARES on the ToolProvider contract (TRIGGER_KEYWORDS /
# ALWAYS_ATTACH) — this layer holds no per-service knowledge.


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
    # Env var holding an endpoint override, for backends whose address isn't
    # fixed (self-hosted). Empty means the endpoint is pinned by the class.
    BASE_URL_ENV: str = ""
    # Hosted vendors authenticate with a key and must fail fast without one.
    # Self-hosted backends (Ollama) accept any credential and set this False.
    REQUIRES_API_KEY: bool = True
    # Per-request timeout (seconds). Hosted vendors answer in seconds, so a
    # tight cap keeps failover snappy; local inference is far slower and
    # raises this (see OllamaAgent).
    REQUEST_TIMEOUT: float = 30.0
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
    # When the window overflows, trim down to this FRACTION of the budget in
    # one step instead of evicting a row per turn. See _history_floor.
    HISTORY_TRIM_TO = 0.5

    # Token-constrained vendors (small TPM) set this True to send only the
    # relevant tools per turn (see _select_tools) instead of the full set.
    SUBSET_TOOLS: bool = False

    def __init__(
        self,
        context_builder: ContextBuilder,
        history: ConversationHistory,
        persona_id: str,
        chat_id: ConversationRef,
        connectors: Optional[list[Connector]] = None,
        persona: Optional[PersonaLike] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        external_tools_provider: Optional[
            Any  # async () -> dict[str, ToolSpec]; see adapters/model/external_mcp.py
        ] = None,
        max_tokens: Optional[int] = None,
        extra_completion_kwargs: Optional[dict[str, Any]] = None,
        supports_vision: Optional[bool] = None,
    ) -> None:
        self._composer = context_builder
        self._history = history
        self._persona_id = persona_id
        self._chat_id = chat_id
        self._connectors = connectors or []
        self._persona = persona
        self._model = model or self.DEFAULT_MODEL
        # No env fallback: the composition root (or eval harness) resolves
        # API_KEY_ENV and passes the key in — settings own the environment.
        self._api_key = api_key
        self._base_url = base_url or self.DEFAULT_BASE_URL
        self._client = None
        self._current_task: Optional[asyncio.Task] = None
        self._external_tools_provider = external_tools_provider
        self._external_tools_loaded = False
        self._max_tokens = max_tokens or None  # 0/None → vendor default
        # Sticky lower bound on replayed history — see _history_floor.
        self._history_floor_id = 0
        # Vision is a property of the MODEL, not the vendor: a self-hosted
        # backend serves whatever was pulled, and gemma4's e4b builds have no
        # vision while its 12b does. Declaring it wrongly means images get
        # sent to a model that cannot see them.
        if supports_vision is not None:
            self.SUPPORTS_VISION = supports_vision
        # Class defaults, overridable per deployment (self-hosted backends run
        # whatever model the operator pulled, and the right knobs differ by
        # model — see OllamaAgent).
        self._extra_kwargs: dict[str, Any] = {
            **self.EXTRA_COMPLETION_KWARGS, **(extra_completion_kwargs or {}),
        }
        self.last_turn_usage: dict[str, Any] = {}
        if not self._api_key and self.REQUIRES_API_KEY:
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
        # Provider-declared routing: base name -> (always_attach, keywords).
        # Providers absent from this map (external MCP servers) opted out of
        # routing and ride every turn.
        self._provider_routing = {
            c.name: (c.ALWAYS_ATTACH, tuple(c.TRIGGER_KEYWORDS))
            for c in self._connectors
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
        ~8.8k before history). So we always include providers that declared
        ALWAYS_ATTACH (memory, schedule) and attach the rest only when the
        message mentions their name or one of their TRIGGER_KEYWORDS —
        provider-declared routing with a safe fallback (providers that
        declared nothing always ride). Non-constrained vendors send
        everything.
        """
        if not self.SUBSET_TOOLS or not self._openai_tools:
            return self._openai_tools
        low = (text or "").lower()

        def _attach(base: Optional[str]) -> bool:
            routing = self._provider_routing.get(base)
            if routing is None:
                return True  # unknown/external provider → always keep
            always, keywords = routing
            if always or not keywords:
                return True
            return base in low or any(k in low for k in keywords)

        tools = [
            t for name, t in self._openai_tool_by_name.items()
            if _attach(self._tool_connector.get(name))
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
                    **self._extra_kwargs,
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
        # The SDK rejects an empty api_key outright, so keyless backends
        # (Ollama) send a placeholder the server ignores.
        kwargs: dict[str, Any] = {"api_key": self._api_key or "no-key-required"}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        # CRITICAL for latency: CascadingAgent IS the retry/failover layer, so
        # the SDK's own retries are redundant AND harmful — on a 429 the SDK
        # would otherwise retry twice with exponential backoff (honoring
        # Retry-After), burning tens of seconds per rate-limited vendor before
        # we ever get to fail over. max_retries=0 makes a busy vendor fail
        # instantly so we advance to the next one immediately. Tight timeout
        # caps a hung request (SDK default is 600s).
        self._client = AsyncOpenAI(
            max_retries=0, timeout=self.REQUEST_TIMEOUT, **kwargs
        )

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

    async def prewarm(self) -> bool:
        """Build this turn's prompt prefix and send it with a 1-token cap, so
        the engine caches it BEFORE a human is waiting on it.

        Local inference pays ~100s to prefill a cold ~13k-token prompt and
        ~0.6s once it's cached. Nothing makes that first prefill cheap — but
        it doesn't have to happen while the user watches a typing indicator.
        Firing it at startup moves the cost off the hot path entirely.

        Only the system prompt + tool schemas are warmed here (the bulk of
        the prefix); a real turn's history and user text still prefill, but
        that's the small tail. Best-effort: any failure is swallowed, since
        this is an optimisation and never correctness.
        """
        if self._client is None:
            await self.start()
        await self._merge_external_tools()
        messages = [
            {"role": "system", "content": self._composer.build()},
            {"role": "user", "content": "."},
        ]
        try:
            await self._client.chat.completions.create(
                model=self._model, messages=messages,
                tools=self._openai_tools, max_tokens=1, **self._extra_kwargs,
            )
            return True
        except Exception as e:
            log.debug("prewarm skipped (%s)", str(e)[:120])
            return False

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

    @staticmethod
    def _is_replayable(row: dict[str, Any]) -> bool:
        meta = row.get("metadata") or {}
        if row["role"] == "summary":
            return False  # always kept, never counted against the window
        if row["role"] == "system" and not meta.get("tool_use"):
            return False
        return row["role"] in ("user", "assistant", "system")

    def _history_floor(self, rows: list[dict[str, Any]]) -> int:
        """Oldest mirror row id that may replay — sticky, and only ever moves
        forward in big steps.

        A plain newest-first budget evicts exactly one old row per turn once
        the window is full. That changes the FIRST replayed message every
        turn, and since the KV cache is only reused for a byte-identical
        prefix, it silently forces a full re-prefill on every single turn —
        measured here as ~53s average on turns that should have cost ~5s.

        Trimming to HISTORY_TRIM_TO of the budget in one jump means the window
        then GROWS by appending (prefix stays identical, cache hits) until it
        overflows again — turning a per-turn cost into a once-every-N-turns
        cost. The cost isn't free, it's amortised: with TRIM_TO=0.5 roughly
        half the window refills before the next trim.
        """
        replayable = [r for r in rows if self._is_replayable(r)]
        in_window = [r for r in replayable
                     if (r.get("id") or 0) >= self._history_floor_id]
        total = sum(len(r["content"]) for r in in_window)
        if total <= self.MAX_HISTORY_CHARS:
            return self._history_floor_id

        target = self.MAX_HISTORY_CHARS * self.HISTORY_TRIM_TO
        acc = 0
        kept_any = False
        new_floor = self._history_floor_id
        for row in reversed(replayable):          # newest first
            rid = row.get("id") or 0
            cost = len(row["content"])
            if kept_any and acc + cost > target:
                new_floor = rid + 1               # this row and older drop out
                break
            # The newest row ALWAYS rides, even if it alone exceeds the
            # target — dropping the turn we are answering would be absurd.
            acc += cost
            kept_any = True
            new_floor = rid
        self._history_floor_id = max(self._history_floor_id, new_floor)
        return self._history_floor_id

    def _assemble_context(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Budget-based replay of the mirror. Newest rows win the budget;
        summary rows always ride along; mirrored tool calls surface as
        inline system notes so this vendor knows what actions were taken
        (possibly by a different vendor).

        The window's lower edge is sticky (see _history_floor) so the replayed
        prefix stays byte-identical between trims — that is what lets a local
        model reuse its KV cache instead of re-reading the whole prompt."""
        kept: list[dict[str, Any]] = []
        budget = self.MAX_HISTORY_CHARS
        floor = self._history_floor(rows)
        rows = [r for r in rows
                if r["role"] == "summary" or (r.get("id") or 0) >= floor]
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
                    **self._extra_kwargs,
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
                # Empty string, NOT a placeholder: CascadingAgent decides what
                # a blank turn means (it fails over). See the note at the
                # final-answer return below.
                return ""

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
                # Final answer. Return the EMPTY STRING when the model produced
                # nothing — never a human-readable placeholder.
                #
                # This used to return "(no response)", which silently defeated
                # CascadingAgent's empty-reply failover: the sentinel is not
                # empty, so a dead turn looked like a successful one and got
                # relayed to the user verbatim while healthy vendors sat idle.
                # Reporting emptiness truthfully is what lets the layer that
                # owns failover actually see it; rendering it for humans is the
                # display layer's job.
                return (msg.content or "").strip()

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
                    result = await spec.handler(
                        args, ToolContext(chat_id=self._chat_id),
                    )
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


class OllamaAgent(ChatCompletionsAgent):
    # Local models served by Ollama's OpenAI-compatible endpoint
    # (http://localhost:11434/v1). Unlike every other vendor here there is no
    # account, no key and no quota — the constraint is the host's RAM/GPU, so
    # `enabled` keys off OLLAMA_ENABLED/OLLAMA_MODEL rather than a key (see
    # runtime/vendors.py). Point OLLAMA_BASE_URL at another machine to use a
    # box on the LAN; inside docker-compose that's host.docker.internal.
    #
    # Default is gemma4:12b — it advertises tools + vision, which is the bar
    # for being useful as this agent's primary. Tool-calling reliability on a
    # 12B local model is below Llama-3.3-70B's, so the hosted vendors are
    # worth keeping in the chain behind it.
    #
    # IMPORTANT: raise num_ctx on the SERVER before using this in anger.
    # Stock Ollama caps it at 4096 no matter what the model supports, and its
    # /v1 layer ignores per-request `options`, so this class cannot fix it
    # from here. A normal turn overflows 4096, and Ollama then drops the
    # OLDEST tokens — i.e. the system prompt. The model keeps answering, just
    # with no persona and no grounding, which reads as "the bot got dumb"
    # rather than as an error. See docs/DEPLOYING.md, "Running against a
    # local model", for the two ways to raise it.
    DEFAULT_MODEL = "gemma4:12b"
    DEFAULT_BASE_URL = "http://localhost:11434/v1"
    API_KEY_ENV = ""          # keyless
    BASE_URL_ENV = "OLLAMA_BASE_URL"
    REQUIRED_ENV: list[str] = []
    REQUIRES_API_KEY = False
    SUPPORTS_VISION = True
    # DELIBERATELY OFF, unlike every metered vendor — and it makes turns
    # ~50x faster, not slower.
    #
    # Subsetting sends a keyword-chosen tool list per turn. Tool schemas are
    # rendered into the PROMPT PREFIX, so a list that changes per message
    # changes the prefix, and llama.cpp can only reuse the KV cache up to the
    # first differing byte. Measured on the same prompt: identical tool list
    # → 0.60s prefill; changed tool list → 41.14s. Live turns were landing at
    # 78-104s for exactly this reason.
    #
    # The trade subsetting exists to make (fewer tokens, at the cost of a
    # varying prefix) is backwards here: local tokens are unmetered, and a
    # STABLE full tool list is prefilled once and then reused for free. The
    # extra schema costs one cold turn; the varying list costs every turn.
    SUBSET_TOOLS = False
    # Thinking is left at the MODEL'S DEFAULT and is deliberately NOT disabled
    # here, because the right answer differs per model and getting it wrong
    # breaks the bot in opposite directions:
    #
    #   gemma4:12b  thinking cost ~16x the output tokens for an identical
    #               answer (98 tok/7.6s vs 6 tok/0.4s), and since thinking is
    #               billed against max_tokens a capped turn could spend the
    #               whole budget reasoning and return EMPTY content. It wants
    #               reasoning_effort=none.
    #   qwen3.5:9b  the OPPOSITE. With a realistic ~55-tool prompt,
    #               reasoning_effort=none produced empty content and NO tool
    #               call; with thinking on it selects the right tool. It needs
    #               its reasoning to pick from a large tool set.
    #
    # So this is per-deployment: set OLLAMA_REASONING_EFFORT (e.g. "none") in
    # the instance .env for models that don't need to think. Unset = leave the
    # model alone, which is the safe default. Ollama ignores per-request
    # `options` but its /v1 layer does honor reasoning_effort.
    EXTRA_COMPLETION_KWARGS: dict[str, Any] = {}
    # NOT a context limit — that is raised on the Ollama server (see
    # docs/DEPLOYING.md). This is purely a LATENCY budget: local prefill runs
    # on the order of 100 tok/s, so every ~100 prompt tokens costs a second of
    # dead time before the first output token appears.
    MAX_HISTORY_CHARS = 12_000  # ≈ 3k tokens ≈ 23s of prefill
    # A local 12B at ~20-40 tok/s needs minutes for a long answer, and a cold
    # first call also pays model load. The hosted-vendor 30s cap would time
    # out mid-generation on nearly every turn and fail over pointlessly.
    REQUEST_TIMEOUT = 300.0


class ChatCompletionsSummarizer(Summarizer):
    """`Summarizer` that runs memory/history compaction through any
    OpenAI-compatible vendor (Gemini/OpenAI/DeepSeek). This is what keeps the
    memory subsystem LLM-agnostic: a Gemini-primary bot summarizes with Gemini,
    not Claude. Reuses the vendor backend's model/base_url/key/extra config so
    there's a single source of truth per vendor.
    """

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None,
                 extra: Optional[dict] = None, timeout: float = 30.0) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._extra = dict(extra or {})
        self._timeout = timeout
        self._client = None  # lazy AsyncOpenAI

    @classmethod
    def for_backend(
        cls,
        backend: type,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> "ChatCompletionsSummarizer":
        """Build from a ChatCompletionsAgent subclass — single source of
        truth for base_url/extra kwargs per vendor. The composition root
        resolves model/api_key/base_url/extra from settings; no env reads here.
        `base_url` and `extra` override the backend defaults (self-hosted
        endpoints, whose right knobs depend on the model in use)."""
        if not api_key and backend.REQUIRES_API_KEY:
            raise RuntimeError(
                f"{backend.__name__} summarizer: no API key configured "
                f"(set {backend.API_KEY_ENV})"
            )
        return cls(
            model=model or backend.DEFAULT_MODEL,
            api_key=api_key,
            base_url=base_url or backend.DEFAULT_BASE_URL,
            extra={**backend.EXTRA_COMPLETION_KWARGS, **(extra or {})},
            timeout=backend.REQUEST_TIMEOUT,
        )

    async def summarize(self, prompt: str, *, deep: bool = False) -> str:
        if self._client is None:
            from openai import AsyncOpenAI
            # Same fast-fail rationale as the agent client (see start()):
            # summarization is best-effort/background, so don't let SDK
            # retries stall it either.
            self._client = AsyncOpenAI(api_key=self._api_key or "no-key-required",
                                       base_url=self._base_url,
                                       max_retries=0, timeout=self._timeout)
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            **self._extra,
        )
        return (resp.choices[0].message.content or "").strip()
