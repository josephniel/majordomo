"""chat.formatting — markdown stripping, chunking, cancel-intent."""
import pytest

from kernel.formatting import _md_to_plain, chunk_for_platform, is_cancel_intent


class TestMdToPlain:
    def test_strips_bold_stars(self):
        assert _md_to_plain("**bold** text") == "bold text"

    def test_strips_bold_underscores(self):
        assert _md_to_plain("__bold__ text") == "bold text"

    def test_strips_inline_code(self):
        assert _md_to_plain("run `ls -la` now") == "run ls -la now"

    def test_strips_fenced_code_keeps_content(self):
        assert _md_to_plain("```python\nx = 1\n```") == "x = 1\n"

    def test_strips_headers(self):
        assert _md_to_plain("# Title\n## Sub\nbody") == "Title\nSub\nbody"

    def test_link_becomes_text_and_url(self):
        assert _md_to_plain("[site](https://x.y)") == "site (https://x.y)"

    def test_keeps_bullets(self):
        assert _md_to_plain("- one\n- two") == "- one\n- two"

    def test_plain_text_unchanged(self):
        assert _md_to_plain("hello world") == "hello world"


class TestChunkForPlatform:
    def test_short_text_single_chunk(self):
        assert chunk_for_platform("hello", 100) == ["hello"]

    def test_empty_text_yields_one_empty_chunk(self):
        assert chunk_for_platform("", 100) == [""]

    def test_exact_limit_is_one_chunk(self):
        assert chunk_for_platform("a" * 100, 100) == ["a" * 100]

    def test_over_limit_splits(self):
        chunks = chunk_for_platform("a" * 250, 100)
        assert [len(c) for c in chunks] == [100, 100, 50]

    def test_reassembles_losslessly(self):
        text = "x" * 999
        assert "".join(chunk_for_platform(text, 100)) == text

    def test_markdown_stripped_before_chunking(self):
        chunks = chunk_for_platform("**hi**", 100)
        assert chunks == ["hi"]


class TestCancelIntent:
    @pytest.mark.parametrize("msg", [
        "cancel", "stop", "abort", "nvm", "never mind", "nevermind",
        "stop it", "cancel that", "please cancel", "ok stop", "Stop!",
        "CANCEL", "stop it now",
    ])
    def test_pure_cancel_messages(self, msg):
        assert is_cancel_intent(msg) is True

    @pytest.mark.parametrize("msg", [
        # Real content that merely CONTAINS a cancel word must not cancel.
        "cancel my subscription",
        "stop the music",
        "can you cancel the meeting tomorrow",
        "how do I abort a git rebase",
        "the show was cancelled",
        "halt and catch fire is a good show",
        # Unrelated messages
        "hello there", "", "   ",
        # 'never' or 'mind' alone are not triggers
        "never", "mind", "never again",
    ])
    def test_non_cancel_messages(self, msg):
        assert is_cancel_intent(msg) is False

    def test_regression_docstring_case(self):
        # The old implementation swallowed this as a cancel (its own
        # docstring claimed otherwise). It is a real request.
        assert is_cancel_intent("cancel my subscription") is False
