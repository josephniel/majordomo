"""Postgres connector — read the company's databases, and write to them twice.

The first tool in this codebase that reaches a database the agent does not own.
Everything else here talks HTTP to a service with its own permission model;
this talks SQL to a server that will do whatever the credential is allowed to
do. The whole design follows from that.

THE FENCE IS ENTIRELY CLIENT-SIDE. The bot connects with a human operator's
own account, so there is no least-privilege role underneath refusing what this
module fails to catch. `_sql.classify_*` is therefore load-bearing rather than
defence-in-depth, and the read path's protection is a session opened READ ONLY
— which is why `_sql` refuses SET and RESET in every spelling. A dedicated role
per environment is the named upgrade path: point the write pool at a write role
and the read pool at a read role, and these guards become the second layer they
ought to be.

A WRITE COSTS TWO TAPS, AND THE FIRST ONE CHANGES NOTHING.

    sql_preview   free. what would this touch? nothing is written.
    sql_apply     tap 1. runs the statement for real inside a transaction that
                  is ALWAYS rolled back, and reports what WOULD change.
    sql_commit    tap 2. re-runs it and commits, or rolls back if the number of
                  affected rows moved since the dry run.

Not one held-open transaction across both taps: that would hold row locks on a
production table for the length of a human's attention span, and the approval
timeout is 300 seconds. Losing the pending-write registry to a restart is safe
by construction — nothing is open server-side, and the failure direction is
always "the write did not happen".

The environment is a routing argument on every call and renders first in the
approval prompt, above a banner this connector computes itself. Approving a
production write while believing it is staging is the worst realistic failure
of this tool, and the prompt is where it would happen.

DELIBERATELY ABSENT:
  * DELETE, DDL, GRANT/REVOKE, SET/RESET, transaction control, upserts —
    refused by `_sql`, in code, not by prompt guidance.
  * Bound parameters for model SQL. The whole statement is model-authored, so
    `$1` moves no risk; it only adds type-coercion failure modes. (The
    catalogue queries below DO bind their arguments — there the model supplies
    a value, not a statement.)
  * A write tool at all, unless the profile opts in. Without
    DATABASE_ALLOW_WRITES the write tools are NOT EMITTED — not present and
    refusing, absent — so `/status` and the prompt both tell the truth and the
    model never proposes what it cannot do.

Secrets per profile (credentials/database/<profile>/secrets.json):
    DATABASE_HOST, DATABASE_PORT, DATABASE_USER, DATABASE_PASSWORD
    DATABASE_ALLOWED_DBS     comma-separated; a call naming anything else is
                             refused and the refusal lists these
    DATABASE_ALLOW_WRITES    "true" to emit the write tools at all
    DATABASE_WRITE_TABLES    comma-separated schema.table the writes may touch
    DATABASE_MAX_WRITE_ROWS  optional; may only LOWER the persona's cap
"""

from __future__ import annotations

import logging
import re
import secrets as secrets_mod
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import asyncpg

from ports import (
    ConversationRef,
    PreviewRefusedError,
    ToolContext,
    ToolResult,
    ToolSpec,
    tool,
)

from ._sql import Analysis, Refusal, classify_read, classify_write

if TYPE_CHECKING:
    from collections.abc import Sequence

log = logging.getLogger(__name__)

CONNECT_TIMEOUT = 10.0
MAX_SQL_CHARS = 8000
# One screenful of evidence. A bigger cap does not help a human decide, and
# does help an injected SELECT move more data into the chat in one call.
PREVIEW_ROWS = 20
CELL_CHARS = 120
OUTPUT_CHARS = 6000

HANDLE_RE = re.compile(r"^w_[a-f0-9]{6,}$")

UNATTENDED = (
    "database writes are refused on unattended turns — a write happens in a "
    "live conversation with the operator or not at all"
)

# Catalogue queries. Fixed text with placeholders: the model supplies a schema
# or table NAME, never a fragment, so there is nothing here for a crafted name
# to reach.
TABLES_SQL = """
SELECT n.nspname || '.' || c.relname AS name,
       CASE c.relkind WHEN 'r' THEN 'table' WHEN 'v' THEN 'view'
            WHEN 'm' THEN 'matview' ELSE c.relkind::text END AS kind,
       c.reltuples::bigint AS approx_rows
  FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
 WHERE c.relkind IN ('r','v','m')
   AND n.nspname NOT IN ('pg_catalog', 'information_schema')
   AND n.nspname NOT LIKE 'pg_toast%'
   AND ($1 = '' OR n.nspname = $1)
   AND c.relname ILIKE $2
 ORDER BY n.nspname, c.relname LIMIT $3
"""

SCHEMAS_SQL = """
SELECT n.nspname AS schema, count(*) AS tables
  FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
 WHERE c.relkind IN ('r','v','m')
   AND n.nspname NOT IN ('pg_catalog', 'information_schema')
   AND n.nspname NOT LIKE 'pg_toast%'
 GROUP BY 1 ORDER BY 2 DESC
"""

COLUMNS_SQL = """
SELECT n.nspname || '.' || c.relname AS "table", a.attname AS column,
       format_type(a.atttypid, a.atttypmod) AS type,
       NOT a.attnotnull AS nullable,
       pg_get_expr(d.adbin, d.adrelid) AS default_value
  FROM pg_attribute a
  JOIN pg_class c ON c.oid = a.attrelid
  JOIN pg_namespace n ON n.oid = c.relnamespace
  LEFT JOIN pg_attrdef d ON d.adrelid = c.oid AND d.adnum = a.attnum
 WHERE ($1 = '' OR n.nspname = $1) AND c.relname = $2
   AND a.attnum > 0 AND NOT a.attisdropped
   AND n.nspname NOT IN ('pg_catalog', 'information_schema')
 ORDER BY n.nspname, a.attnum
"""

# Kept out of the f-strings that build tool descriptions: this is prose, but it
# reads like a query and there is no reason to make a reader check.
QUERY_DOC = (
    "Runs inside a READ ONLY transaction with a statement timeout, so it "
    "cannot write and cannot hang. Semicolon-separated statements, "
    "data-modifying CTEs, row-locking selects (FOR UPDATE/SHARE), SET/RESET "
    "and all DDL are refused before execution. Column names come from "
    "sql_describe. Args: target, database, sql, limit (default 50), offset."
)

_CAP_PREFIX = "SELECT * FROM ("
_CAP_SUFFIX = ") AS _capped LIMIT $1 OFFSET $2"


def redact(text: str, password: str) -> str:
    """Strip a password out of anything on its way to a human or a log."""
    return text.replace(password, "***") if password else text


@dataclass(frozen=True)
class PendingWrite:
    """A dry run waiting for its second tap.

    Holds the statement itself so `sql_commit` can take a handle and nothing
    else — the SQL cannot be swapped between the approval that showed it and
    the approval that commits it.
    """

    handle: str
    database: str
    table: str
    sql: str
    preview_sql: str | None
    reason: str
    affected: int
    chat_id: ConversationRef | None
    created: float


@dataclass(frozen=True)
class Profile:
    """One environment's connection settings, and what it may touch."""

    name: str
    environment: str
    host: str
    port: int
    user: str
    password: str
    databases: tuple[str, ...]
    write_tables: frozenset[str]
    allow_writes: bool
    max_write_rows: int
    statement_timeout_ms: int
    max_rows_returned: int
    pending_ttl_seconds: int

    def banner(self, database: str) -> str:
        mark = "⛔ PRODUCTION" if self.environment == "production" else "staging"
        return f"{mark} — {self.host} / {database}"


class DatabaseClient:
    """Pools, execution, and the pending-write registry for one profile.

    One pool per (database, mode). The read pool opens every connection with
    `default_transaction_read_only`, and every read additionally runs inside an
    explicit READ ONLY transaction: a session default is a setting, and a
    setting is the kind of thing a statement could change.
    """

    def __init__(self, profile: Profile, persona_id: str) -> None:
        self.profile = profile
        self._persona_id = persona_id
        self._pools: dict[tuple[str, bool], asyncpg.Pool] = {}
        self._pending: dict[str, PendingWrite] = {}

    # ---- connections ----

    def _dsn(self, database: str) -> str:
        p = self.profile
        return (
            f"postgresql://{quote(p.user, safe='')}:{quote(p.password, safe='')}"
            f"@{p.host}:{p.port}/{quote(database, safe='')}"
        )

    async def _pool(self, database: str, *, writable: bool) -> asyncpg.Pool:
        key = (database, writable)
        existing = self._pools.get(key)
        if existing is not None:
            return existing
        p = self.profile
        settings = {
            "statement_timeout": str(p.statement_timeout_ms),
            "idle_in_transaction_session_timeout": "30000",
            "application_name": (
                f"majordomo/{self._persona_id}/{p.name}/{'rw' if writable else 'ro'}"
            ),
        }
        if not writable:
            settings["default_transaction_read_only"] = "on"
        pool: asyncpg.Pool = await asyncpg.create_pool(
            self._dsn(database),
            min_size=0,
            max_size=2,
            timeout=CONNECT_TIMEOUT,
            server_settings=settings,
        )
        self._pools[key] = pool
        return pool

    async def close(self) -> None:
        for pool in self._pools.values():
            await pool.close()
        self._pools.clear()

    # ---- execution ----

    async def read(
        self, database: str, sql: str, *params: object
    ) -> list[asyncpg.Record]:
        """Run one statement inside an explicit READ ONLY transaction.

        Always through the extended protocol, never `execute` with a bare
        string: the simple query protocol allows several statements in one
        round trip and the extended one refuses them. That refusal is a fence
        no parser can provide.
        """
        pool = await self._pool(database, writable=False)
        async with pool.acquire() as conn, conn.transaction(readonly=True):
            rows: list[asyncpg.Record] = await conn.fetch(sql, *params)
            return rows

    async def dry_run(self, database: str, sql: str) -> list[asyncpg.Record]:
        """Execute for real, capture the result, then always roll back."""
        pool = await self._pool(database, writable=True)
        async with pool.acquire() as conn:
            transaction = conn.transaction()
            await transaction.start()
            try:
                rows: list[asyncpg.Record] = await conn.fetch(sql + " RETURNING *")
                return rows
            finally:
                await transaction.rollback()

    async def commit(self, database: str, sql: str, expected: int) -> tuple[int, str]:
        """Re-run and commit — unless the affected count moved since the dry run."""
        pool = await self._pool(database, writable=True)
        async with pool.acquire() as conn:
            transaction = conn.transaction()
            await transaction.start()
            try:
                rows = await conn.fetch(sql + " RETURNING *")
            except Exception:
                await transaction.rollback()
                raise
            if len(rows) != expected:
                await transaction.rollback()
                return len(rows), (
                    f"rolled back: this would now affect {len(rows)} row(s), not "
                    f"the {expected} you approved — someone else changed the data. "
                    "Nothing was written; run sql_apply again to see where things "
                    "stand."
                )
            await transaction.commit()
            return len(rows), ""

    # ---- the pending-write registry ----

    def remember(self, write: PendingWrite) -> None:
        self._pending[write.handle] = write

    def outstanding(self, chat_id: ConversationRef | None) -> PendingWrite | None:
        self._expire()
        for write in self._pending.values():
            if write.chat_id == chat_id:
                return write
        return None

    def take(self, handle: str, chat_id: ConversationRef | None) -> PendingWrite | str:
        """Look up a handle, or say why it cannot be used."""
        self._expire()
        write = self._pending.get(handle)
        if write is None:
            return (
                f"no pending write for {handle} — it may have expired, already "
                "been committed, or been lost to a restart. Nothing was written; "
                "run sql_apply again."
            )
        if write.chat_id != chat_id:
            return "that handle belongs to a different conversation"
        return write

    def forget(self, handle: str) -> None:
        self._pending.pop(handle, None)

    def _expire(self) -> None:
        cutoff = time.monotonic() - self.profile.pending_ttl_seconds
        for handle in [h for h, w in self._pending.items() if w.created < cutoff]:
            del self._pending[handle]


# --------------------------------------------------------------------------
# Rendering and argument checking
# --------------------------------------------------------------------------


def _cell(value: Any) -> str:
    text = "NULL" if value is None else str(value)
    text = " ".join(text.split())
    return text if len(text) <= CELL_CHARS else text[:CELL_CHARS] + "…"


def _render(rows: Sequence[asyncpg.Record], limit: int = PREVIEW_ROWS) -> str:
    if not rows:
        return "(no rows)"
    lines = [
        "  " + "  ".join(f"{k}={_cell(v)}" for k, v in row.items())
        for row in rows[:limit]
    ]
    if len(rows) > limit:
        lines.append(f"  … (+{len(rows) - limit} more rows not shown)")
    text = "\n".join(lines)
    if len(text) > OUTPUT_CHARS:
        text = text[:OUTPUT_CHARS] + "\n  … (output truncated)"
    return text


def _refuse(message: str) -> ToolResult:
    return ToolResult.error(message)


def _bounded(sql: str) -> str:
    """Wrap a classified SELECT so a row cap binds regardless of its own LIMIT.

    The numbers ride as parameters; the only text concatenated is the statement
    the classifier already accepted.
    """
    return _CAP_PREFIX + sql.rstrip().rstrip(";") + _CAP_SUFFIX


def _route(
    client: DatabaseClient, args: dict[str, Any]
) -> tuple[str, str] | ToolResult:
    """Check the two routing arguments every tool carries."""
    profile = client.profile
    target = str(args.get("target", "")).strip()
    if target != profile.environment:
        return _refuse(
            f"this tool is bound to {profile.environment!r}; you asked for "
            f"{target!r}. Use the {profile.environment} tools instead — nothing "
            "was executed."
        )
    database = str(args.get("database", "")).strip()
    if database not in profile.databases:
        allowed = ", ".join(profile.databases)
        return _refuse(
            f"{database!r} is not a database on {target}. Available: {allowed}"
        )
    return target, database


def _statement(args: dict[str, Any]) -> str | ToolResult:
    sql = str(args.get("sql", "")).strip()
    if not sql:
        return _refuse("no SQL given")
    if len(sql) > MAX_SQL_CHARS:
        return _refuse(f"statement is {len(sql)} chars; the limit is {MAX_SQL_CHARS}")
    return sql


def _connection_error(client: DatabaseClient, err: Exception) -> str:
    detail = redact(str(err)[:200], client.profile.password)
    return (
        f"could not reach {client.profile.host}:{client.profile.port} — "
        f"{type(err).__name__}: {detail}. Both environments sit on private VPC "
        "addresses, so check the VPN. Nothing was executed."
    )


def _driver_error(client: DatabaseClient, err: Exception, prefix: str) -> ToolResult:
    """Turn a driver exception into something an operator can act on."""
    if isinstance(err, (OSError, TimeoutError)):
        return _refuse(_connection_error(client, err))
    detail = redact(str(err)[:300], client.profile.password)
    return _refuse(f"{prefix}{type(err).__name__}: {detail}")


async def _attempt(
    client: DatabaseClient, coro: Any, prefix: str
) -> Any | ToolResult:
    """Await a database call, or return the refusal that explains the failure."""
    try:
        return await coro
    except (OSError, TimeoutError, asyncpg.PostgresError) as err:
        return _driver_error(client, err, prefix)


async def _guarded(
    client: DatabaseClient,
    database: str,
    sql: str,
    label: str,
    limit: int = PREVIEW_ROWS,
    params: tuple[object, ...] = (),
) -> ToolResult:
    try:
        rows = await client.read(database, sql, *params)
    except (OSError, TimeoutError) as err:
        return _refuse(_connection_error(client, err))
    except asyncpg.PostgresError as err:
        detail = redact(str(err)[:300], client.profile.password)
        return _refuse(f"{label}: {type(err).__name__}: {detail}")
    return ToolResult.ok(f"{len(rows)} row(s)\n{_render(rows, limit)}")


async def _blast_radius(client: DatabaseClient, database: str, verdict: Analysis) -> str:
    """Describe what a write would touch, or refuse it for touching too much."""
    cap = client.profile.max_write_rows
    if verdict.literal_rows is not None:
        return f"{verdict.sql}\ninserts {verdict.literal_rows} literal row(s)"
    if verdict.preview_sql is None:
        return f"{verdict.sql}\n(no preview available)"
    rows = await client.read(database, _bounded(verdict.preview_sql), cap + 1, 0)
    if len(rows) > cap:
        raise PreviewRefusedError(
            f"this would affect more than {cap} rows — refused. Narrow the "
            "WHERE clause, or have the operator raise the cap for this profile. "
            "Nothing was executed."
        )
    return f"{verdict.sql}\nmatches {len(rows)} row(s) (cap {cap}):\n{_render(rows)}"


def _schema_for(profile: Profile, extra: dict[str, Any]) -> dict[str, Any]:
    """Build a tool schema with the two routing arguments already in place."""
    properties: dict[str, Any] = {
        "target": {
            "type": "string",
            "enum": [profile.environment],
            "description": (
                f"Must be {profile.environment!r}. Named explicitly so the "
                "approval prompt can show which environment this touches."
            ),
        },
        "database": {
            "type": "string",
            "enum": list(profile.databases),
            "description": "Which database on that host.",
        },
    }
    properties.update(extra)
    return {
        "type": "object",
        "properties": properties,
        "required": ["target", "database"],
    }


def _handle_schema(profile: Profile) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "target": {"type": "string", "enum": [profile.environment]},
            "handle": {"type": "string", "pattern": HANDLE_RE.pattern},
        },
        "required": ["target", "handle"],
    }


# --------------------------------------------------------------------------
# Read tools
# --------------------------------------------------------------------------


def _tables_tool(client: DatabaseClient) -> ToolSpec:
    profile = client.profile

    @tool(
        "sql_tables",
        f"List tables and views on {profile.environment}, as schema.name. "
        "Searches EVERY schema unless you name one — these databases keep "
        "almost nothing in `public`. There are hundreds of tables, so use "
        "`filter`. Call with no filter to see what schemas exist. Args: "
        "target, database, schema (optional), filter (case-insensitive "
        "substring on the table name), limit (default 60).",
        _schema_for(
            profile,
            {
                "schema": {"type": "string"},
                "filter": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            },
        ),
    )
    async def handler(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        routed = _route(client, args)
        if isinstance(routed, ToolResult):
            return routed
        _, database = routed
        schema = str(args.get("schema") or "")
        needle = str(args.get("filter") or "")
        limit = min(int(args.get("limit") or 60), 200)
        result = await _guarded(
            client,
            database,
            TABLES_SQL,
            f"tables in {schema or 'every schema'}",
            limit,
            params=(schema, f"%{needle}%" if needle else "%", limit),
        )
        if result.is_error or not result.text.startswith("0 row(s)"):
            return result
        # Nothing matched. Say which schemas DO hold tables rather than let
        # the model conclude the database is empty.
        return await _guarded(
            client, database, SCHEMAS_SQL,
            f"no table matched {needle!r}"
            + (f" in schema {schema!r}" if schema else "")
            + "; schemas that do hold tables",
        )

    return handler


def _describe_tool(client: DatabaseClient) -> ToolSpec:
    profile = client.profile
    env = profile.environment

    @tool(
        "sql_describe",
        f"Describe one table on {env}: columns, types, nullability, defaults. "
        "Call this before writing a query — it is where column names come "
        "from. A bare name is looked up in every schema; qualify it "
        "(schema.table) when the name is not unique. Args: target, database, "
        "table.",
        _schema_for(profile, {"table": {"type": "string"}}),
    )
    async def handler(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        routed = _route(client, args)
        if isinstance(routed, ToolResult):
            return routed
        _, database = routed
        raw = str(args.get("table", "")).strip()
        schema, _, name = raw.rpartition(".")
        result = await _guarded(
            client, database, COLUMNS_SQL, raw, 200, params=(schema, name)
        )
        if result.is_error:
            return result
        if result.text.startswith("0 row(s)"):
            return _refuse(
                f"no table named {raw!r} in {database}. Find it with "
                f"sql_tables(target={env!r}, database={database!r}, "
                f"filter={name!r})."
            )
        example = " ".join(("SELECT * FROM", f"{schema}.{name}", "LIMIT 10"))
        return ToolResult.ok(
            f"{result.text}\n\nExample: sql_query(target={env!r}, "
            f"database={database!r}, sql={example!r})"
        )

    return handler


def _query_tool(client: DatabaseClient) -> ToolSpec:
    profile = client.profile

    @tool(
        "sql_query",
        f"Run ONE read-only query on {profile.environment}. " + QUERY_DOC,
        _schema_for(
            profile,
            {
                "sql": {"type": "string", "maxLength": MAX_SQL_CHARS},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                "offset": {"type": "integer", "minimum": 0},
            },
        ),
    )
    async def handler(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        routed = _route(client, args)
        if isinstance(routed, ToolResult):
            return routed
        _, database = routed
        sql = _statement(args)
        if isinstance(sql, ToolResult):
            return sql
        verdict = classify_read(sql)
        if isinstance(verdict, Refusal):
            return _refuse(f"refused ({verdict.code}): {verdict.message}")
        limit = min(int(args.get("limit") or 50), profile.max_rows_returned)
        offset = max(int(args.get("offset") or 0), 0)
        return await _guarded(
            client,
            database,
            _bounded(verdict.sql),
            "query",
            limit,
            params=(limit, offset),
        )

    return handler


def _preview_tool(client: DatabaseClient) -> ToolSpec:
    profile = client.profile

    @tool(
        "sql_preview",
        "Show what an INSERT or UPDATE WOULD touch. NOTHING IS WRITTEN. Runs "
        "the equivalent SELECT and reports the exact rows and the count. Call "
        "this first and tell the user what you found, before asking them to "
        "approve anything — sql_apply computes the same thing independently, "
        "so a surprise there is one you should have raised here. Args: target, "
        "database, sql.",
        _schema_for(profile, {"sql": {"type": "string", "maxLength": MAX_SQL_CHARS}}),
    )
    async def handler(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        routed = _route(client, args)
        if isinstance(routed, ToolResult):
            return routed
        _, database = routed
        sql = _statement(args)
        if isinstance(sql, ToolResult):
            return sql
        verdict = classify_write(sql, profile.write_tables)
        if isinstance(verdict, Refusal):
            return _refuse(f"refused ({verdict.code}): {verdict.message}")
        try:
            summary = await _blast_radius(client, database, verdict)
        except PreviewRefusedError as refused:
            return _refuse(str(refused))
        except (OSError, TimeoutError, asyncpg.PostgresError) as err:
            return _driver_error(client, err, "preview failed: ")
        return ToolResult.ok(f"{profile.banner(database)}\n{summary}\nNothing written.")

    return handler


def read_tools(client: DatabaseClient) -> list[ToolSpec]:
    """Assemble the read surface: find your way around, then read."""
    return [
        _tables_tool(client),
        _describe_tool(client),
        _query_tool(client),
        _preview_tool(client),
    ]


# --------------------------------------------------------------------------
# Write tools — two taps, and the first one changes nothing
# --------------------------------------------------------------------------


def _checked_write(
    client: DatabaseClient, args: dict[str, Any]
) -> tuple[str, Analysis] | ToolResult:
    """Route, bound and classify a write in one step."""
    routed = _route(client, args)
    if isinstance(routed, ToolResult):
        return routed
    _, database = routed
    sql = _statement(args)
    if isinstance(sql, ToolResult):
        return sql
    verdict = classify_write(sql, client.profile.write_tables)
    if isinstance(verdict, Refusal):
        return _refuse(f"refused ({verdict.code}): {verdict.message}")
    return database, verdict


def _apply_tool(client: DatabaseClient) -> ToolSpec:
    profile = client.profile
    env = profile.environment
    tables = ", ".join(sorted(profile.write_tables)) or "(none)"

    async def preview(args: dict[str, Any], _ctx: ToolContext) -> str:
        """Compute what the operator sees before tap 1.

        Runs against the live database, so the rows and the banner are this
        connector's word rather than the model's.
        """
        checked = _checked_write(client, args)
        if isinstance(checked, ToolResult):
            raise PreviewRefusedError(checked.text)
        database, verdict = checked
        summary = await _blast_radius(client, database, verdict)
        return (
            f"{profile.banner(database)}\n{summary}\n"
            "this is a DRY RUN — it will be rolled back. sql_commit is a "
            "separate tap."
        )

    @tool(
        "sql_apply",
        f"DRY RUN one INSERT or UPDATE on {env}. The statement really executes, "
        "inside a transaction that is ALWAYS ROLLED BACK, so it tells you what "
        "would change without changing it. Nothing persists until you call "
        "sql_commit with the handle this returns — a second, separate "
        f"approval. UPDATE must have a WHERE. Writable tables: {tables}. "
        "DELETE, DDL and upserts are refused in code. Args: target, database, "
        "sql, reason (one line: why, and on whose instruction).",
        _schema_for(
            profile,
            {
                "sql": {"type": "string", "maxLength": MAX_SQL_CHARS},
                "reason": {"type": "string", "maxLength": 300},
            },
        ),
        approval_preview=preview,
    )
    async def handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.background:
            return _refuse(UNATTENDED)
        checked = _checked_write(client, args)
        if isinstance(checked, ToolResult):
            return checked
        database, verdict = checked

        outstanding = client.outstanding(ctx.chat_id)
        if outstanding is not None:
            return _refuse(
                f"there is already a pending write in this chat "
                f"({outstanding.handle}, {outstanding.affected} row(s) on "
                f"{outstanding.table}). Commit it with sql_commit or drop it "
                "with sql_discard before starting another."
            )
        rows = await _attempt(
            client,
            client.dry_run(database, verdict.sql),
            "the statement failed and was rolled back: ",
        )
        if isinstance(rows, ToolResult):
            return rows
        if len(rows) > profile.max_write_rows:
            return _refuse(
                f"this affects {len(rows)} rows, over the cap of "
                f"{profile.max_write_rows}. Rolled back, nothing written."
            )
        return _remember(client, args, database, verdict, rows, ctx)

    return handler


def _remember(
    client: DatabaseClient,
    args: dict[str, Any],
    database: str,
    verdict: Analysis,
    rows: list[asyncpg.Record],
    ctx: ToolContext,
) -> ToolResult:
    """Record the dry run and hand back the handle its commit will need."""
    handle = f"w_{secrets_mod.token_hex(3)}"
    client.remember(
        PendingWrite(
            handle=handle,
            database=database,
            table=verdict.table or "?",
            sql=verdict.sql,
            preview_sql=verdict.preview_sql,
            reason=str(args.get("reason", "")),
            affected=len(rows),
            chat_id=ctx.chat_id,
            created=time.monotonic(),
        )
    )
    return ToolResult.ok(
        f"{client.profile.banner(database)}\nDRY RUN — rolled back, nothing "
        f"written.\n{len(rows)} row(s) would change:\n{_render(rows)}\n\n"
        f"handle={handle} — call sql_commit to make it real (a second approval)."
    )


def _commit_tool(client: DatabaseClient) -> ToolSpec:
    profile = client.profile

    async def preview(args: dict[str, Any], ctx: ToolContext) -> str:
        """Compute what the operator sees before tap 2 — recomputed, not replayed."""
        found = client.take(str(args.get("handle", "")).strip(), ctx.chat_id)
        if isinstance(found, str):
            raise PreviewRefusedError(found)
        if found.preview_sql:
            rows = await client.read(
                found.database,
                _bounded(found.preview_sql),
                profile.max_write_rows + 1,
                0,
            )
            current = f"{len(rows)} row(s) match it right now"
        else:
            current = "an insert — nothing to re-check"
        return (
            f"{profile.banner(found.database)}\nCOMMIT — this one persists.\n"
            f"{found.sql}\n"
            f"the dry run affected {found.affected} row(s); {current}\n"
            f"reason: {found.reason}"
        )

    @tool(
        "sql_commit",
        "Commit a dry run. Takes ONLY the handle sql_apply returned — the SQL "
        "is read from the pending write, so it cannot change between the "
        "approval that showed it and the one that commits it. If the number of "
        "affected rows has moved since the dry run this rolls back instead. "
        "Handles expire and do not survive a restart; either way nothing was "
        "written. Args: target, handle.",
        _handle_schema(profile),
        approval_preview=preview,
    )
    async def handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.background:
            return _refuse(UNATTENDED)
        handle = str(args.get("handle", "")).strip()
        if not HANDLE_RE.match(handle):
            return _refuse(f"{handle!r} is not a handle sql_apply would have issued")
        found = client.take(handle, ctx.chat_id)
        if isinstance(found, str):
            return _refuse(found)
        outcome = await _attempt(
            client,
            client.commit(found.database, found.sql, found.affected),
            "the commit failed and was rolled back: ",
        )
        if isinstance(outcome, ToolResult):
            return outcome
        affected, drift = outcome
        if drift:
            return _refuse(drift)
        client.forget(handle)
        return ToolResult.ok(
            f"{profile.banner(found.database)}\nCOMMITTED — {affected} row(s) "
            f"changed in {found.table}."
        )

    return handler


def _discard_tool(client: DatabaseClient) -> ToolSpec:
    profile = client.profile

    @tool(
        "sql_discard",
        "Drop a pending write without committing it. Nothing was ever "
        "persisted, so this only frees the slot. Args: target, handle.",
        _handle_schema(profile),
    )
    async def handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        handle = str(args.get("handle", "")).strip()
        found = client.take(handle, ctx.chat_id)
        if isinstance(found, str):
            return _refuse(found)
        client.forget(handle)
        return ToolResult.ok(
            f"dropped {handle} ({found.affected} row(s) on {found.table} never "
            "written)."
        )

    return handler


def write_tools(client: DatabaseClient) -> list[ToolSpec]:
    """Assemble the write surface. Only called when the profile opts in."""
    return [_apply_tool(client), _commit_tool(client), _discard_tool(client)]
