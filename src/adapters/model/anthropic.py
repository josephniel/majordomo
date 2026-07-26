"""Anthropic-backed agent — full Claude Agent SDK with MCP tools.

This is the primary agent in the fallback chain. Auth is one of:
  - ANTHROPIC_API_KEY in process env, OR
  - subscription auth via the bundled Claude CLI (no env var needed).

When the SDK signals a usage limit (rate-limit / overload / quota), this
agent re-raises as `UsageLimitError` so CascadingAgent can rotate.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    CLIConnectionError,
    ProcessError,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
)
from claude_agent_sdk import (
    tool as _claude_sdk_tool,
)

from ports import (
    Connector,
    ConversationRef,
    ServiceCatalog,
    ToolContext,
    ToolSpec,
    as_tool_result,
)

from .base import (
    Agent,
    Attachment,
    ContextBuilder,
    PersonaLike,
    Summarizer,
    ToolUseCallback,
    UsageLimitError,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

log = logging.getLogger(__name__)

# claude.ai-hosted MCP connectors that the bundled CLI exposes under
# subscription auth. We don't want the model picking these — they'd shadow
# our own in-process / external connectors with the same service names.
CLAUDE_AI_CONNECTOR_TOOLS = [
    "mcp__claude_ai_ClickUp__authenticate",
    "mcp__claude_ai_ClickUp__complete_authentication",
    "mcp__claude_ai_Gmail__authenticate",
    "mcp__claude_ai_Gmail__complete_authentication",
    "mcp__claude_ai_Google_Calendar__authenticate",
    "mcp__claude_ai_Google_Calendar__complete_authentication",
    "mcp__claude_ai_Google_Drive__authenticate",
    "mcp__claude_ai_Google_Drive__complete_authentication",
    "mcp__claude_ai_Microsoft_365__authenticate",
    "mcp__claude_ai_Microsoft_365__complete_authentication",
]

# Substrings we look for in raised exceptions to classify "usage limit hit"
# vs. other transient errors. Matched against str(exception).lower().
_USAGE_LIMIT_HINTS = (
    "rate_limit",
    "rate limit",
    "overloaded",
    "quota",
    "usage limit",
    "429",
    "too many requests",
    "model is overloaded",
)


def _is_usage_limit(exc: BaseException) -> bool:
    # The Claude Agent SDK surfaces rate limits as CLI ProcessErrors rather than
    # typed exceptions, so we match hint substrings — ProcessError already folds
    # the CLI stderr into its message, so the underlying rate-limit text is
    # visible in str(exc). Walk the cause/context chain so a wrapped error is
    # still classified.
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        msg = (str(cur) or "").lower()
        if any(hint in msg for hint in _USAGE_LIMIT_HINTS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


# Compaction (memory + chat history) reuses the bundled Claude CLI's
# subscription auth instead of going to api.anthropic.com directly. That
# means callers don't need ANTHROPIC_API_KEY — the same credential that
# powers the main chat also covers compaction.
DEFAULT_COMPACTION_MODEL = "claude-haiku-4-5"
DEFAULT_COMPACTION_FALLBACK_MODEL = "claude-sonnet-5"


async def summarize_with_subscription_auth(
    prompt: str,
    primary_model: str = DEFAULT_COMPACTION_MODEL,
    fallback_model: str | None = DEFAULT_COMPACTION_FALLBACK_MODEL,
) -> str:
    """One-off completion for compaction / summarization tasks.

    Routes through claude_agent_sdk so the bundled CLI's subscription auth
    is used (no ANTHROPIC_API_KEY required). Tries `primary_model` first;
    if that errors out (e.g. model not exposed to subscription), retries
    with `fallback_model`. Returns the assistant's text or empty string
    if both attempts fail.
    """
    try:
        return await _sdk_one_shot(prompt, primary_model)
    except Exception as e:
        if fallback_model is None or fallback_model == primary_model:
            log.warning(
                "subscription summarize failed on %s and no fallback configured: %s",
                primary_model, e,
            )
            return ""
        log.info(
            "subscription summarize failed on %s (%s); retrying with %s",
            primary_model, e, fallback_model,
        )
    try:
        return await _sdk_one_shot(prompt, fallback_model)
    except Exception:
        log.exception("subscription summarize failed on fallback %s", fallback_model)
        return ""


async def _sdk_one_shot(prompt: str, model: str) -> str:
    """Minimal stateless completion via ClaudeSDKClient.

    No tools, no MCP, no system prompt — just send `prompt` as the user turn and read the reply.
    Client is created and torn down per call (~200ms cold start).
    """
    opts = ClaudeAgentOptions(
        model=model,
        mcp_servers={},
        allowed_tools=[],
        disallowed_tools=list(CLAUDE_AI_CONNECTOR_TOOLS),
        permission_mode="bypassPermissions",
        setting_sources=[],
        tools=[],
    )
    async with ClaudeSDKClient(options=opts) as client:
        await client.query(prompt)
        chunks: list[str] = []
        actual_model = None
        usage = None
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                actual_model = getattr(msg, "model", None) or actual_model
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        chunks.append(block.text)
            elif isinstance(msg, ResultMessage):
                actual_model = getattr(msg, "model", None) or actual_model
                usage = getattr(msg, "usage", None)
        # Observability: prove which model actually served this background
        # call (requested vs what the server reports back).
        log.info(
            "background summarize: requested=%s served=%s usage=%s",
            model, actual_model or "(not reported)", usage,
        )
        return "".join(chunks).strip()


class SubscriptionAuthSummarizer(Summarizer):
    """Concrete `Summarizer` that runs through the bundled Claude CLI's
    subscription auth — no API key needed.

    Default models: Haiku-4.5 for normal, Sonnet-4.6 for deep. Both are
    customizable so a persona can override (e.g. point deep at Opus).
    """

    def __init__(
        self,
        primary_model: str = DEFAULT_COMPACTION_MODEL,
        deep_model: str = DEFAULT_COMPACTION_FALLBACK_MODEL,
    ) -> None:
        self._primary = primary_model
        self._deep = deep_model

    async def summarize(self, prompt: str, *, deep: bool = False) -> str:
        primary = self._deep if deep else self._primary
        # Normal mode: if the cheaper primary isn't available to this
        # subscription, fall through to the deep model. Deep mode: no
        # further fallback — caller asked for capability, give it one shot.
        fallback = None if deep else self._deep
        return await summarize_with_subscription_auth(
            prompt, primary_model=primary, fallback_model=fallback,
        )


def _to_claude_sdk_tool(spec: ToolSpec, ctx: ToolContext):
    """Translate a vendor-neutral ToolSpec into a claude_agent_sdk @tool.

    The SDK's `@tool` decorator wraps an async handler with metadata it uses
    when emitting the in-process MCP server. We unbox the ToolSpec back into
    that wrapper here so connectors stay vendor-neutral. `json_schema()` is
    the normalized form (full JSON Schema with required/descriptions/enums);
    the SDK passes schema dicts through to MCP unchanged. Handler results
    (ToolResult, or legacy dicts from external MCP servers) become the MCP
    wire shape HERE — providers never build it. `ctx` is the dispatching
    agent's chat scope, bound per mount since each agent is chat-scoped.
    """
    @_claude_sdk_tool(spec.name, spec.description, spec.json_schema())
    async def _wrapped(args: dict[str, Any]):
        return _mcp_content(await spec.handler(args, ctx))
    return _wrapped


def _mcp_content(raw: Any) -> dict[str, Any]:
    """MCP wire shape from a handler result.

    Lives here, at the one edge that speaks MCP, rather than in `ports`.
    It was in the contracts package for a while, which put an Anthropic wire
    format in the module every faculty imports — the exact leak that made
    "vendor-neutral tools" only nearly true. `as_tool_result` is the neutral
    half and stays in ports; this is the translation, and translation belongs
    to the translator.
    """
    r = as_tool_result(raw)
    out: dict[str, Any] = {"content": [{"type": "text", "text": r.text}]}
    if r.is_error:
        out["isError"] = True
    return out


class AnthropicOptionsBuilder:
    """Builds ClaudeAgentOptions from connectors + profiles + persona.

    Pure composition — no IO at construction time. `build()` reads
    connectors.yaml and assembles the system prompt + MCP server map.
    """

    def __init__(
        self,
        context_builder: ContextBuilder,
        config: ServiceCatalog,
        connectors: list[Connector],
        persona: PersonaLike,
        model: str | None = None,
        max_turns: int | None = None,
        max_output_tokens: int | None = None,
        default_model: str | None = None,
    ) -> None:
        self._composer = context_builder
        self._config = config
        self._connectors = connectors
        self._persona = persona
        # Explicit override > persona pin > composition-root default
        # (settings.claude_model). No env reads here — RuntimeSettings is
        # the only place environment becomes config.
        self._model = model or persona.model or default_model or "claude-sonnet-5"
        self._max_turns = max_turns or None  # 0/None → uncapped
        self._max_output_tokens = max_output_tokens or None

    @property
    def model(self) -> str:
        return self._model

    def build(
        self,
        resume_session_id: str | None = None,
        chat_id: ConversationRef | None = None,
    ) -> ClaudeAgentOptions:
        enabled = self._config.load_enabled()
        ctx = ToolContext(chat_id=chat_id)
        mcp_servers: dict[str, Any] = {}
        allowed_tools: list[str] = []

        # In-process MCPs. Connectors return vendor-neutral ToolSpec objects;
        # we wrap each as a claude_agent_sdk @tool here so they're consumable
        # by ClaudeSDKClient. Translation lives in this layer; connectors
        # don't import claude_agent_sdk.
        for c in self._connectors:
            allowed_for_c = self._persona.allowed_tool_names(c)
            if allowed_for_c == []:
                continue
            for server_name, specs in c.builtin_servers().items():
                filtered = self._filter_tool_specs(specs, allowed_for_c)
                if not filtered:
                    continue
                sdk_tools = [_to_claude_sdk_tool(spec, ctx) for spec in filtered]
                mcp_servers[server_name] = create_sdk_mcp_server(
                    name=server_name, version="1.0.0", tools=sdk_tools,
                )
                # The SDK's allow-list wants full MCP names; the mcp__ naming
                # convention is THIS vendor's concern, derived here from the
                # same persona-filtered specs that were just mounted.
                allowed_tools.extend(
                    f"mcp__{server_name}__{spec.name}" for spec in filtered
                )

        # External MCPs (subprocess stdio) from connectors.yaml.
        for i in enabled:
            if i.name in mcp_servers:
                continue
            allowed_for_c = self._allowed_for_profile(i.name)
            if allowed_for_c == []:
                continue
            mcp_servers[i.name] = {
                "type": "stdio",
                "command": i.command,
                "args": i.args,
                "env": i.env,
            }
            for tool_name in i.allowed_tools:
                if allowed_for_c is None or tool_name in allowed_for_c:
                    allowed_tools.append(f"mcp__{i.name}__{tool_name}")

        # The CLI reads its output cap from the environment; env here is
        # additive to the inherited subprocess environment.
        env: dict[str, str] = {}
        if self._max_output_tokens:
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(self._max_output_tokens)

        return ClaudeAgentOptions(
            system_prompt=self._composer.build(),
            model=self._model,
            mcp_servers=mcp_servers,
            allowed_tools=allowed_tools,
            disallowed_tools=list(CLAUDE_AI_CONNECTOR_TOOLS),
            permission_mode="bypassPermissions",
            setting_sources=[],
            tools=[],
            resume=resume_session_id,
            max_turns=self._max_turns,
            env=env,
        )

    def _allowed_for_profile(self, profile_name: str) -> list[str] | None:
        for c in self._connectors:
            if c.owns_profile(profile_name):
                return self._persona.allowed_tool_names(c)
        return None

    @staticmethod
    def _filter_tool_specs(
        specs: list[ToolSpec],
        allowed_names: list[str] | None,
    ) -> list[ToolSpec]:
        """Filter ToolSpec list by name. None = all allowed; [] = none."""
        if allowed_names is None:
            return list(specs)
        if not allowed_names:
            return []
        return [s for s in specs if s.name in allowed_names]


class AnthropicAgent(Agent):
    """One persistent Claude conversation per chat (session-based)."""

    REQUIRED_ENV: ClassVar[list[str]] = []  # Subscription-auth fallback handles missing key.
    USES_SERVER_SIDE_HISTORY = True  # Claude sessions live CLI-side.

    def __init__(
        self,
        options_builder: AnthropicOptionsBuilder,
        session_id: str | None = None,
        chat_id: ConversationRef | None = None,
    ) -> None:
        self._options_builder = options_builder
        self._session_id: str | None = session_id
        # The chat this agent serves — bound into every mounted tool's
        # ToolContext so handlers know their scope without ambient state.
        self._chat_id = chat_id
        self._client: ClaudeSDKClient | None = None
        self.last_turn_usage: dict[str, Any] = {}

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def model_name(self) -> str:
        return self._options_builder.model

    async def start(self) -> None:
        try:
            await self._open(self._session_id)
        except (ProcessError, CLIConnectionError) as e:
            # A stale/expired resume id makes the CLI abort at startup
            # ("No conversation found with session ID ..."). Rather than fail
            # the whole turn, drop the session and start a fresh conversation.
            if not self._session_id:
                raise
            log.warning(
                "resume of session %s failed (%s); starting a fresh session",
                self._session_id,
                e,
            )
            await self._discard_client()
            self._session_id = None
            await self._open(None)

    async def reset_session(self) -> None:
        """Abandon the resumed session and open a fresh one.

        The caller (CascadingAgent rotation) reseeds context from the mirror; without this, a
        resumed session replays the entire conversation as input tokens on every turn, forever.
        """
        await self._discard_client()
        self._session_id = None
        await self._open(None)

    async def _open(self, session_id: str | None) -> None:
        opts = self._options_builder.build(session_id, chat_id=self._chat_id)
        self._client = ClaudeSDKClient(options=opts)
        await self._client.__aenter__()

    async def _discard_client(self) -> None:
        if self._client is None:
            return
        try:
            await self._client.__aexit__(None, None, None)
        except Exception:
            pass
        finally:
            self._client = None

    async def stop(self) -> None:
        if self._client is not None:
            try:
                await self._client.__aexit__(None, None, None)
            finally:
                self._client = None

    async def interrupt(self) -> None:
        if self._client is None:
            return
        with contextlib.suppress(Exception):
            await self._client.interrupt()

    async def send(
        self,
        text: str,
        on_tool_use: ToolUseCallback | None = None,
        attachments: list[Attachment] | None = None,
        current_row_id: int | None = None,  # server-side history: unused
    ) -> str:
        if self._client is None:
            await self.start()
        assert self._client is not None

        try:
            if attachments:
                await self._client.query(self._stream_multimodal(text, attachments))
            else:
                await self._client.query(text)

            self.last_turn_usage = {}
            parts: list[str] = []
            async for msg in self._client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            parts.append(block.text)
                        elif isinstance(block, ToolUseBlock) and on_tool_use is not None:
                            with contextlib.suppress(Exception):
                                await on_tool_use(block.name, dict(block.input or {}))
                elif isinstance(msg, ResultMessage):
                    self._session_id = msg.session_id
                    self._capture_usage(msg)
            # Empty string, not a placeholder — same contract as the
            # chat-completions agents: a blank turn must stay visibly blank so
            # CascadingAgent's empty-reply failover can see it.
            return "".join(parts).strip()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if _is_usage_limit(e):
                raise UsageLimitError(
                    f"Anthropic usage limit hit: {e}"
                ) from e
            raise

    def _capture_usage(self, msg: ResultMessage) -> None:
        """Pull token usage off the ResultMessage (previously discarded).

        Shape varies slightly across SDK versions — read defensively.
        """
        try:
            usage = getattr(msg, "usage", None) or {}
            if not isinstance(usage, dict):
                usage = dict(usage)
            self.last_turn_usage = {
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
            }
            cost = getattr(msg, "total_cost_usd", None)
            if cost is not None:
                self.last_turn_usage["cost_usd"] = cost
        except Exception:
            self.last_turn_usage = {}

    @staticmethod
    def _attachment_to_content_block(att: Attachment) -> dict[str, Any] | None:
        if att.media_type.startswith("image/"):
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": att.media_type,
                    "data": base64.b64encode(att.data).decode("ascii"),
                },
            }
        if att.media_type == "application/pdf":
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64.b64encode(att.data).decode("ascii"),
                },
            }
        if att.media_type.startswith("text/"):
            try:
                decoded = att.data.decode("utf-8", errors="replace")
                return {
                    "type": "text",
                    "text": f"[Attached file content]\n\n{decoded}",
                }
            except Exception:
                return None
        return {
            "type": "text",
            "text": f"[Unsupported attachment of type {att.media_type}]",
        }

    @staticmethod
    async def _stream_multimodal(
        text: str, attachments: list[Attachment]
    ) -> AsyncIterator[dict]:
        content: list[dict] = []
        if text:
            content.append({"type": "text", "text": text})
        for att in attachments:
            block = AnthropicAgent._attachment_to_content_block(att)
            if block is not None:
                content.append(block)
        yield {
            "type": "user",
            "message": {"role": "user", "content": content},
        }
