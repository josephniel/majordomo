"""Statement classification for the database connector.

Pure and I/O-free: parse a model-authored statement, decide whether it may run,
and derive the SELECT that shows what a write would touch. No connection, no
config, no logging — so the whole fence is unit-testable without a database.

WHY A PARSER AND NOT A REGEX. `UPDATE t SET x=1 -- WHERE id=5` looks fenced to
any regex that searches for the word WHERE, and is an unfenced UPDATE to
Postgres. The asymmetry runs one way: a comment cannot hide a statement from a
parser, but it can trivially hide one from a pattern match. Against production
with a human's own credentials there is no server-side net under this, so the
classifier is load-bearing rather than defence-in-depth.

THE SHAPE IS ALLOW-LIST FIRST. A statement is refused unless its root node is
one of a handful of permitted types; the named deny-list below exists only so
the refusal can say WHICH rule fired. That ordering matters because sqlglot
funnels everything it does not model into `exp.Command` — verified: SET ROLE,
SET SESSION AUTHORIZATION, RESET, PREPARE/EXECUTE, DO $$…$$, VACUUM, CREATE
ROLE and ALTER ROLE all land there — so unknown syntax fails closed without
this module needing to enumerate it.

DELIBERATELY ABSENT, and refused in code no matter what anyone approves:
  * DELETE — Phase 1 covers SELECT/INSERT/UPDATE only.
  * INSERT ... ON CONFLICT DO UPDATE. It parses as a plain Insert but can
    modify rows that already exist, so the row count and the preview would
    both understate what it does.
  * All DDL (CREATE/ALTER/DROP/TRUNCATE) and GRANT/REVOKE.
  * SET / RESET in every spelling. This one is load-bearing, not hygiene: the
    read path's protection is a session opened read-only, and a single
    `SET TRANSACTION READ WRITE` would unlock it. Verified that spelling parses
    as `exp.Set` while `SET ROLE` parses as `exp.Command`, so both paths are
    covered.
  * Transaction control (BEGIN/COMMIT/ROLLBACK/SAVEPOINT). The connector owns
    transaction boundaries; a statement that could close one could commit a
    dry run. Note SAVEPOINT parses as `exp.Alias`, not a transaction node —
    which is precisely why the allow-list, not the deny-list, is what refuses
    it.

KNOWN LIMITS, stated because a guard nobody distrusts is a guard nobody checks:
  * sqlglot's Postgres dialect is not libpg_query. A statement it mis-parses
    could be mis-classified. Unparseable input is refused, and unmodelled input
    becomes Command and is refused — but a *differential* divergence, something
    sqlglot reads as a benign SELECT that Postgres executes otherwise, is the
    residual risk and nothing here catches it.
  * A WHERE clause is not a bound. `WHERE id > 0` is fenced and matches every
    row. The row cap, applied by the caller against the derived preview, is
    what bounds a write.
  * A SELECT can still write, through a VOLATILE function. Only the read-only
    transaction stops that, which is why the session-level setting is not
    optional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import sqlglot
from sqlglot import exp

__all__ = [
    "READ_REASONS",
    "WRITE_REASONS",
    "Analysis",
    "Refusal",
    "classify_read",
    "classify_write",
]

# Statement roots that are never acceptable on any path. Checked before the
# allow-list purely so the refusal names the rule the operator broke; the
# allow-list is what actually decides.
_FORBIDDEN: tuple[tuple[type[exp.Expr], str], ...] = (
    (exp.Command, "unsupported or unrecognised statement"),
    (exp.Set, "SET"),
    (exp.Grant, "GRANT"),
    (exp.Revoke, "REVOKE"),
    (exp.Create, "CREATE"),
    (exp.Alter, "ALTER"),
    (exp.Drop, "DROP"),
    (exp.TruncateTable, "TRUNCATE"),
    (exp.Delete, "DELETE"),
    (exp.Merge, "MERGE"),
    (exp.Transaction, "BEGIN"),
    (exp.Commit, "COMMIT"),
    (exp.Rollback, "ROLLBACK"),
    (exp.Copy, "COPY"),
)

# DML anywhere in a read's tree. A CTE is the hole this closes: verified that
# `WITH x AS (UPDATE t SET a=1 RETURNING *) SELECT * FROM x` parses with a
# Select at the root, so checking the root type alone would pass it.
_DML: tuple[type[exp.Expr], ...] = (exp.Update, exp.Insert, exp.Delete, exp.Merge)

ReadReason = Literal[
    "unparseable",
    "multiple_statements",
    "forbidden_statement",
    "not_a_read",
    "dml_in_read",
    "lock_in_read",
    "select_into",
]
WriteReason = Literal[
    "unparseable",
    "multiple_statements",
    "forbidden_statement",
    "not_a_write",
    "cte_in_write",
    "nested_dml",
    "no_where",
    "trivial_where",
    "no_table",
    "table_not_allowed",
    "upsert",
    "returning_in_write",
]
READ_REASONS: frozenset[str] = frozenset(
    (
        "unparseable",
        "multiple_statements",
        "forbidden_statement",
        "not_a_read",
        "dml_in_read",
        "lock_in_read",
        "select_into",
    )
)
WRITE_REASONS: frozenset[str] = frozenset(
    (
        "unparseable",
        "multiple_statements",
        "forbidden_statement",
        "not_a_write",
        "cte_in_write",
        "nested_dml",
        "no_where",
        "trivial_where",
        "no_table",
        "table_not_allowed",
        "upsert",
        "returning_in_write",
    )
)


@dataclass(frozen=True)
class Refusal:
    """A statement that will not run, and the rule that stopped it.

    `code` is asserted by the tests rather than the message: a rule that starts
    firing for the wrong reason is a rule that will later stop firing.
    """

    code: str
    message: str


@dataclass(frozen=True)
class Analysis:
    """An accepted statement, normalised, with what the caller needs next.

    `preview_sql` is the SELECT that shows which rows a write would touch. It
    is None for a read (there is nothing to preview) and for an INSERT whose
    rows are literal, where `literal_rows` already gives the exact count with
    no round trip.
    """

    kind: Literal["select", "insert", "update"]
    sql: str
    table: str | None = None
    preview_sql: str | None = None
    literal_rows: int | None = None


def _parse_one(sql: str) -> exp.Expr | Refusal:
    """Parse exactly one statement, or return the rule that stopped it."""
    text = sql.strip()
    if not text:
        return Refusal("unparseable", "empty statement")
    try:
        statements = sqlglot.parse(text, dialect="postgres")
    except Exception as err:  # sqlglot raises several unrelated types
        return Refusal("unparseable", f"could not parse this SQL: {err}")

    real: list[exp.Expr] = [s for s in statements if s is not None]
    if not real:
        return Refusal("unparseable", "no statement found")
    if len(real) > 1:
        return Refusal(
            "multiple_statements",
            f"{len(real)} statements in one call — send exactly one; "
            "semicolon-separated statements are refused",
        )
    return real[0]


def _forbidden(root: exp.Expr) -> Refusal | None:
    """Name the deny-list rule this statement breaks, if it breaks one."""
    for node_type, label in _FORBIDDEN:
        if isinstance(root, node_type):
            return Refusal(
                "forbidden_statement",
                f"{label} is refused by this tool and cannot be approved",
            )
    return None


def _nested_dml(root: exp.Expr) -> exp.Expr | None:
    """Find DML below the root — the data-modifying CTE hole."""
    for node in root.walk():
        if isinstance(node, _DML) and node is not root:
            return node
    return None


def _read_violation(root: exp.Expr) -> Refusal | None:
    """Check everything that disqualifies a statement from the read path."""
    forbidden = _forbidden(root)
    if forbidden is not None:
        return forbidden
    if not isinstance(root, (exp.Select, exp.Union, exp.Subquery)):
        return Refusal(
            "not_a_read",
            f"only SELECT is allowed here, got {type(root).__name__.upper()}",
        )
    nested = _nested_dml(root)
    if nested is not None:
        return Refusal(
            "dml_in_read",
            f"this SELECT contains a nested {type(nested).__name__.upper()} "
            "(a data-modifying CTE) — refused on the read path",
        )
    # Walk rather than check the root's args: a lock inside a CTE
    # (`WITH c AS (SELECT * FROM t FOR UPDATE) SELECT * FROM c`) leaves the
    # root's own `locks` empty.
    if any(isinstance(n, exp.Lock) for n in root.walk()):
        return Refusal(
            "lock_in_read", "SELECT ... FOR UPDATE/SHARE takes row locks — refused"
        )
    if root.args.get("into") is not None:
        return Refusal("select_into", "SELECT ... INTO creates a table — refused")
    return None


def classify_read(sql: str) -> Analysis | Refusal:
    """Accept exactly one statement that can only read."""
    parsed = _parse_one(sql)
    if isinstance(parsed, Refusal):
        return parsed
    violation = _read_violation(parsed)
    if violation is not None:
        return violation
    return Analysis(kind="select", sql=parsed.sql(dialect="postgres"))


def _where_is_trivial(where: exp.Expr) -> bool:
    """Report whether a WHERE fences nothing.

    A predicate that names no column cannot select between rows: it is either
    true for all of them or false for all of them, so as a fence it is a
    decoration. That covers `WHERE true`, `WHERE 1=1`, `WHERE 'a'='a'` and the
    compound forms (`WHERE 1=1 AND 1=1`) in one rule instead of enumerating
    shapes — and it also refuses `WHERE random() < 0.5`, which fences nothing
    predictable and would touch a different set of rows on the commit than it
    showed on the dry run.

    Anything naming a column is treated as a real fence, however wide:
    `WHERE id > 0` matches every row and is still the row cap's problem, not
    this function's.
    """
    condition = where.this
    if condition is None:
        return True
    return not any(isinstance(n, exp.Column) for n in condition.walk())


def _qualified(table: exp.Table) -> str:
    schema = table.db or "public"
    return f"{schema}.{table.name}"


def _write_shape_violation(root: exp.Expr) -> Refusal | None:
    """Check the statement is a bare INSERT/UPDATE with no DML hidden in it."""
    forbidden = _forbidden(root)
    if forbidden is not None:
        return forbidden
    if not isinstance(root, (exp.Update, exp.Insert)):
        return Refusal(
            "not_a_write",
            f"only INSERT and UPDATE are allowed here, got "
            f"{type(root).__name__.upper()}",
        )
    # A CTE on the write path buys nothing and hides DML, so it is refused
    # outright rather than analysed. Detected by node type, not by an args key:
    # sqlglot spells that key "with_" on an Update and "with" elsewhere, and a
    # key rename between versions would silently reopen this hole.
    if any(isinstance(n, exp.With) for n in root.walk()):
        return Refusal("cte_in_write", "a WITH clause on a write statement is refused")
    nested = _nested_dml(root)
    if nested is not None:
        return Refusal(
            "nested_dml",
            f"nested {type(nested).__name__.upper()} inside a write — refused",
        )
    return _write_clause_violation(root)


def _write_clause_violation(root: exp.Expr) -> Refusal | None:
    """Check the clauses a write carries, as opposed to the shape it has."""
    # ON CONFLICT DO UPDATE mutates rows that already exist while parsing as a
    # plain Insert, so the literal-VALUES row count would understate the blast
    # radius and the preview would show nothing at all.
    if any(isinstance(n, exp.OnConflict) for n in root.walk()):
        return Refusal(
            "upsert",
            "INSERT ... ON CONFLICT can modify existing rows — refused; write "
            "the UPDATE and the INSERT as separate statements",
        )
    # The connector appends its own RETURNING * to capture the dry run. A
    # statement that already carries one would produce two.
    if root.args.get("returning") is not None:
        return Refusal(
            "returning_in_write",
            "drop the RETURNING clause — this tool adds its own to show you "
            "what the write would change",
        )
    return None


def _resolve_table(
    table_node: exp.Table | None, allowed_tables: frozenset[str]
) -> str | Refusal:
    """Return the qualified target table, or the rule that disallows it."""
    if table_node is None:
        return Refusal("no_table", "could not determine the target table")
    table = _qualified(table_node)
    if table not in allowed_tables:
        allowed = (
            ", ".join(sorted(allowed_tables)) or "(none — this profile is read-only)"
        )
        return Refusal(
            "table_not_allowed",
            f"{table} is not on this profile's write allow-list. Allowed: {allowed}",
        )
    return table


def _update_where_violation(root: exp.Update) -> Refusal | None:
    """Check the UPDATE is fenced by a WHERE that actually excludes rows."""
    where = root.args.get("where")
    if where is None:
        return Refusal(
            "no_where",
            "an UPDATE with no WHERE would rewrite every row — refused, and "
            "this refusal cannot be approved away",
        )
    if _where_is_trivial(where):
        return Refusal(
            "trivial_where",
            f"{where.sql(dialect='postgres')} matches every row — refused",
        )
    return None


def classify_write(sql: str, allowed_tables: frozenset[str]) -> Analysis | Refusal:
    """Accept exactly one INSERT or UPDATE against an allow-listed table."""
    parsed = _parse_one(sql)
    if isinstance(parsed, Refusal):
        return parsed
    shape = _write_shape_violation(parsed)
    if shape is not None:
        return shape

    table = _resolve_table(parsed.find(exp.Table), allowed_tables)
    if isinstance(table, Refusal):
        return table

    if isinstance(parsed, exp.Update):
        return _update_analysis(parsed, table)
    if isinstance(parsed, exp.Insert):
        return _insert_analysis(parsed, table)
    return Refusal("not_a_write", "unreachable: shape check already ran")


def _update_analysis(root: exp.Update, table: str) -> Analysis | Refusal:
    """Build the analysis for a fenced UPDATE, including its preview SELECT."""
    violation = _update_where_violation(root)
    if violation is not None:
        return violation
    where = root.args["where"]
    preview = exp.select("*").from_(table).where(where.this.copy())
    return Analysis(
        kind="update",
        sql=root.sql(dialect="postgres"),
        table=table,
        preview_sql=preview.sql(dialect="postgres"),
    )


def _insert_analysis(root: exp.Insert, table: str) -> Analysis:
    """Build the analysis for an INSERT.

    Literal VALUES give an exact row count with no round trip; an
    INSERT ... SELECT previews through its source SELECT instead.
    """
    source = root.expression
    literal_rows = len(source.expressions) if isinstance(source, exp.Values) else None
    preview_sql = (
        source.sql(dialect="postgres")
        if isinstance(source, (exp.Select, exp.Union))
        else None
    )
    return Analysis(
        kind="insert",
        sql=root.sql(dialect="postgres"),
        table=table,
        preview_sql=preview_sql,
        literal_rows=literal_rows,
    )
