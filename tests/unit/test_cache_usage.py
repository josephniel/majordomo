"""Prompt-cache token capture.

`input_tokens` counts UNCACHED input only, so a turn with a warm prefix
reports a value near zero while really sending ~10k tokens — turn_log was
recording `input_tokens: 2` for real turns. Read as "we barely send
anything", that number invites exactly the wrong optimisation. These cover
the two counters that say which it is.
"""
from types import SimpleNamespace

from adapters.model.anthropic import _cache_tokens


class TestClaudeCapture:
    def test_per_model_breakdown_is_preferred(self):
        """model_usage is the CLI's own breakdown and passes camelCase through."""
        msg = SimpleNamespace(model_usage={
            "claude-opus-5": {
                "cacheReadInputTokens": 9000, "cacheCreationInputTokens": 700,
            },
        })
        assert _cache_tokens({}, msg) == {
            "cache_read_tokens": 9000, "cache_write_tokens": 700,
        }

    def test_several_models_in_one_turn_are_summed(self):
        """A turn that failed over mid-flight bills against both."""
        msg = SimpleNamespace(model_usage={
            "claude-opus-5": {"cacheReadInputTokens": 100, "cacheCreationInputTokens": 10},
            "claude-haiku-4-5": {"cacheReadInputTokens": 50, "cacheCreationInputTokens": 5},
        })
        assert _cache_tokens({}, msg) == {
            "cache_read_tokens": 150, "cache_write_tokens": 15,
        }

    def test_falls_back_to_the_flat_snake_case_usage(self):
        msg = SimpleNamespace(model_usage=None)
        usage = {"cache_read_input_tokens": 8000, "cache_creation_input_tokens": 0}
        assert _cache_tokens(usage, msg) == {
            "cache_read_tokens": 8000, "cache_write_tokens": 0,
        }

    def test_an_sdk_reporting_neither_leaves_the_fields_absent(self):
        """NOT zero. "Not measured" and "no cache hits" must not look alike —
        averaging a fabricated zero into the hit rate would understate it."""
        msg = SimpleNamespace(model_usage=None)
        assert _cache_tokens({"input_tokens": 5}, msg) == {}

    def test_a_partial_model_entry_does_not_explode(self):
        msg = SimpleNamespace(model_usage={"m": {"cacheReadInputTokens": None}})
        assert _cache_tokens({}, msg) == {
            "cache_read_tokens": 0, "cache_write_tokens": 0,
        }


class TestTheStatusLine:
    def test_it_reports_a_hit_rate(self):
        from kernel.commands import _cache_line
        line = _cache_line({
            "cache_read_tokens": 9000, "cache_write_tokens": 700, "input_tokens": 300,
        })
        assert "90% of input served from cache" in line

    def test_it_says_nothing_when_nothing_was_measured(self):
        from kernel.commands import _cache_line
        assert _cache_line({"input_tokens": 300}) == ""

    def test_it_survives_an_empty_day(self):
        from kernel.commands import _cache_line
        assert _cache_line({}) == ""
