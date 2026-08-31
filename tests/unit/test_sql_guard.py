"""The SQL fence for the database connector.

Every case asserts the REFUSAL CODE, not merely that something failed. A rule
that starts firing for the wrong reason is a rule that will later stop firing
without anyone noticing, and this fence is the only thing standing between a
model-authored statement and production with a human's own credentials.
"""

from __future__ import annotations

import pytest

from adapters.tools._sql import (
    READ_REASONS,
    WRITE_REASONS,
    Analysis,
    Refusal,
    classify_read,
    classify_write,
)

ALLOWED = frozenset({"public.t", "public.loan_applications"})


def _code(result: Analysis | Refusal) -> str:
    return "accept" if isinstance(result, Analysis) else result.code


# --------------------------------------------------------------------------
# Read path
# --------------------------------------------------------------------------

# Statement control, privilege and DDL verbs. `SET`/`RESET` are here for a
# reason that is easy to lose: reads run in a session opened read-only, and one
# SET would unlock it, making that guard theatre.
FORBIDDEN_ANYWHERE = [
    "SET ROLE postgres",
    "SET SESSION AUTHORIZATION admin",
    "SET TRANSACTION READ WRITE",
    "RESET ALL",
    "BEGIN",
    "COMMIT",
    "ROLLBACK",
    "PREPARE p AS SELECT 1",
    "EXECUTE p",
    "DO $$ BEGIN NULL; END $$",
    "COPY t TO PROGRAM 'sh'",
    "VACUUM FULL t",
    "GRANT ALL ON t TO x",
    "REVOKE ALL ON t FROM x",
    "CREATE ROLE r",
    "ALTER ROLE r SUPERUSER",
    "CREATE INDEX i ON t(a)",
    "ALTER TABLE t ADD COLUMN c int",
    "DROP TABLE t",
    "TRUNCATE t",
    "DELETE FROM t WHERE id = 1",
    "EXPLAIN ANALYZE UPDATE t SET x = 1",
]


@pytest.mark.parametrize("sql", FORBIDDEN_ANYWHERE)
def test_forbidden_verbs_are_refused_on_the_read_path(sql: str) -> None:
    assert _code(classify_read(sql)) == "forbidden_statement"


@pytest.mark.parametrize("sql", FORBIDDEN_ANYWHERE)
def test_forbidden_verbs_are_refused_on_the_write_path(sql: str) -> None:
    assert _code(classify_write(sql, ALLOWED)) == "forbidden_statement"


def test_savepoint_is_refused_by_the_allow_list_not_the_deny_list() -> None:
    """SAVEPOINT parses as an Alias, not a transaction node.

    It is caught because the root is not a permitted type — which is the whole
    argument for ordering the allow-list first.
    """
    assert _code(classify_read("SAVEPOINT s")) == "not_a_read"
    assert _code(classify_write("SAVEPOINT s", ALLOWED)) == "not_a_write"


@pytest.mark.parametrize(
    ("sql", "code"),
    [
        ("SELECT 1; DROP TABLE t", "multiple_statements"),
        ("", "unparseable"),
        ("   ", "unparseable"),
        ("NOT SQL AT ALL", "unparseable"),
        # A data-modifying CTE keeps a Select at the root, so only a walk of
        # the whole tree catches it.
        ("WITH x AS (UPDATE t SET a=1 RETURNING *) SELECT * FROM x", "dml_in_read"),
        ("WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x", "dml_in_read"),
        ("SELECT * FROM t FOR UPDATE", "lock_in_read"),
        ("SELECT * FROM t FOR SHARE", "lock_in_read"),
        ("SELECT * INTO backup FROM t", "select_into"),
        ("UPDATE t SET x = 1 WHERE id = 1", "not_a_read"),
        ("INSERT INTO t (a) VALUES (1)", "not_a_read"),
    ],
)
def test_read_refusals(sql: str, code: str) -> None:
    assert _code(classify_read(sql)) == code


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a, b FROM t WHERE id = 5",
        "SELECT count(*) FROM a JOIN b ON a.id = b.a_id",
        "WITH c AS (SELECT 1) SELECT * FROM c",
        "SELECT * FROM t ORDER BY id DESC LIMIT 10",
    ],
)
def test_reads_accepted(sql: str) -> None:
    result = classify_read(sql)
    assert isinstance(result, Analysis)
    assert result.kind == "select"


# --------------------------------------------------------------------------
# Write path
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "code"),
    [
        ("UPDATE public.t SET x = 1", "no_where"),
        # The case a regex gets wrong: the comment hides the WHERE from a
        # pattern match, and Postgres runs an unfenced UPDATE.
        ("UPDATE public.t SET x = 1 -- WHERE id = 5", "no_where"),
        ("UPDATE public.t SET x = 1 WHERE true", "trivial_where"),
        ("UPDATE public.t SET x = 1 WHERE 1 = 1", "trivial_where"),
        ("WITH c AS (SELECT 1) UPDATE public.t SET x=1 WHERE id=1", "cte_in_write"),
        ("WITH c AS (SELECT 1) INSERT INTO public.t (a) VALUES (1)", "cte_in_write"),
        ("UPDATE public.nope SET x = 1 WHERE id = 1", "table_not_allowed"),
        ("INSERT INTO public.nope (a) VALUES (1)", "table_not_allowed"),
        ("SELECT * FROM t", "not_a_write"),
    ],
)
def test_write_refusals(sql: str, code: str) -> None:
    assert _code(classify_write(sql, ALLOWED)) == code


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE public.t SET x = 1 WHERE 1 = 1 AND 1 = 1",
        "UPDATE public.t SET x = 1 WHERE 'a' = 'a'",
        # Fences nothing predictable, and would touch a different set of rows
        # on the commit than it showed on the dry run.
        "UPDATE public.t SET x = 1 WHERE random() < 0.5",
    ],
)
def test_a_column_free_predicate_is_not_a_fence(sql: str) -> None:
    assert _code(classify_write(sql, ALLOWED)) == "trivial_where"


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO public.t (a) VALUES (1) ON CONFLICT (a) DO UPDATE SET a = 2",
        "INSERT INTO public.t (a) VALUES (1) ON CONFLICT DO NOTHING",
    ],
)
def test_upsert_is_refused(sql: str) -> None:
    """ON CONFLICT parses as a plain Insert but can modify existing rows."""
    assert _code(classify_write(sql, ALLOWED)) == "upsert"


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE public.t SET x = 1 WHERE id = 1 RETURNING *",
        "INSERT INTO public.t (a) VALUES (1) RETURNING id",
    ],
)
def test_user_supplied_returning_is_refused(sql: str) -> None:
    """The connector appends its own RETURNING * to capture the dry run."""
    assert _code(classify_write(sql, ALLOWED)) == "returning_in_write"


def test_a_lock_inside_a_cte_is_caught() -> None:
    """The root's own `locks` arg is empty when the lock sits in a CTE."""
    sql = "WITH c AS (SELECT * FROM t FOR UPDATE) SELECT * FROM c"
    assert _code(classify_read(sql)) == "lock_in_read"


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE public.t SET x = 1 WHERE id = 5",
        "UPDATE public.t SET x = 1 WHERE id IN (SELECT id FROM o WHERE flag)",
        "INSERT INTO public.t (a) VALUES (1)",
    ],
)
def test_real_fences_are_not_over_refused(sql: str) -> None:
    assert _code(classify_write(sql, ALLOWED)) == "accept"


def test_a_wide_but_real_fence_is_accepted() -> None:
    """`WHERE id > 0` matches everything and is still a fence.

    The row cap bounds a write, not the WHERE clause — so this must not be
    refused here, or the caller's cap would never be exercised.
    """
    assert _code(classify_write("UPDATE public.t SET x=1 WHERE id > 0", ALLOWED)) == (
        "accept"
    )


def test_table_refusal_names_the_allowed_tables() -> None:
    result = classify_write("UPDATE public.nope SET x=1 WHERE id=1", ALLOWED)
    assert isinstance(result, Refusal)
    assert "public.t" in result.message
    assert "public.loan_applications" in result.message


def test_empty_allow_list_refuses_every_write_and_says_why() -> None:
    result = classify_write("UPDATE public.t SET x=1 WHERE id=1", frozenset())
    assert isinstance(result, Refusal)
    assert result.code == "table_not_allowed"
    assert "read-only" in result.message


def test_unqualified_table_defaults_to_public() -> None:
    assert _code(classify_write("UPDATE t SET x=1 WHERE id=1", ALLOWED)) == "accept"


# --------------------------------------------------------------------------
# Preview derivation — what the operator is shown before approving
# --------------------------------------------------------------------------


def test_update_preview_is_the_equivalent_select() -> None:
    result = classify_write("UPDATE public.t SET x=1 WHERE id = 88214", ALLOWED)
    assert isinstance(result, Analysis)
    assert result.kind == "update"
    assert result.table == "public.t"
    assert result.preview_sql == "SELECT * FROM public.t WHERE id = 88214"


def test_literal_insert_counts_rows_without_a_round_trip() -> None:
    result = classify_write(
        "INSERT INTO public.t (a, b) VALUES (1, 'x'), (2, 'y')", ALLOWED
    )
    assert isinstance(result, Analysis)
    assert result.literal_rows == 2
    assert result.preview_sql is None


def test_insert_select_previews_through_its_source() -> None:
    result = classify_write(
        "INSERT INTO public.t (a) SELECT a FROM src WHERE id < 10", ALLOWED
    )
    assert isinstance(result, Analysis)
    assert result.literal_rows is None
    assert result.preview_sql == "SELECT a FROM src WHERE id < 10"


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------


def test_every_refusal_code_is_declared() -> None:
    """No refusal may carry a code the module does not publish."""
    samples = [
        *FORBIDDEN_ANYWHERE,
        "SELECT 1; DROP TABLE t",
        "",
        "SELECT * FROM t FOR UPDATE",
        "SELECT * INTO backup FROM t",
        "WITH x AS (UPDATE t SET a=1 RETURNING *) SELECT * FROM x",
        "UPDATE public.t SET x = 1",
        "UPDATE public.t SET x = 1 WHERE true",
        "UPDATE public.nope SET x = 1 WHERE id = 1",
        "SAVEPOINT s",
        "INSERT INTO public.t (a) VALUES (1) ON CONFLICT DO NOTHING",
        "UPDATE public.t SET x = 1 WHERE id = 1 RETURNING *",
        "UPDATE public.t SET x = 1 WHERE random() < 0.5",
    ]
    for sql in samples:
        read = classify_read(sql)
        if isinstance(read, Refusal):
            assert read.code in READ_REASONS, f"{read.code} undeclared for {sql!r}"
        write = classify_write(sql, ALLOWED)
        if isinstance(write, Refusal):
            assert write.code in WRITE_REASONS, f"{write.code} undeclared for {sql!r}"


def test_refusals_carry_a_message() -> None:
    result = classify_write("UPDATE public.t SET x = 1", ALLOWED)
    assert isinstance(result, Refusal)
    assert result.message.strip()
    assert "cannot be approved away" in result.message
