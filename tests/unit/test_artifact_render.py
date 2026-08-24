"""artifact_render — anchors, diff coloring, and the escaping that matters."""
from domain.artifact_render import anchor_for, render_artifact


class TestAnchors:
    def test_finding_headings_anchor_by_tag(self):
        assert anchor_for("F3 — proof names no observable") == "f3"
        assert anchor_for("f12: scope creep") == "f12"

    def test_plain_headings_slugify(self):
        assert anchor_for("API contract") == "api-contract"

    def test_headings_carry_ids_in_the_page(self):
        page = render_artifact("t", "## F1 — thing\n\n## Data model\n")
        assert 'id="f1"' in page
        assert 'id="data-model"' in page


class TestDiffRendering:
    def test_diff_fence_colors_lines(self):
        page = render_artifact("t", "```diff\n+new\n-old\n@@ -1 +1 @@\nctx\n```\n")
        assert '<span class="add">+new</span>' in page
        assert '<span class="del">-old</span>' in page
        assert '<span class="hunk">@@ -1 +1 @@</span>' in page

    def test_file_header_lines_are_not_colored(self):
        page = render_artifact("t", "```diff\n+++ b/x.py\n--- a/x.py\n```\n")
        assert 'class="add">+++' not in page
        assert 'class="del">---' not in page

    def test_diff_content_is_escaped(self):
        page = render_artifact("t", "```diff\n+<script>alert(1)</script>\n```\n")
        assert "<script>alert(1)" not in page
        assert "&lt;script&gt;" in page


class TestSafety:
    def test_raw_html_in_markdown_stays_escaped(self):
        page = render_artifact("t", "<img src=x onerror=alert(1)>\n")
        assert "<img" not in page

    def test_title_is_escaped(self):
        page = render_artifact("<b>MR</b>", "hi")
        assert "<b>MR</b>" not in page
        assert "&lt;b&gt;MR&lt;/b&gt;" in page

    def test_page_declares_noindex(self):
        assert 'name="robots" content="noindex' in render_artifact("t", "x")


class TestChrome:
    def test_updated_stamp_shown_when_given(self):
        assert "updated 2026-08-24" in render_artifact("t", "x", updated="2026-08-24")

    def test_comment_box_posts_to_relative_comment_path(self):
        page = render_artifact("t", "x")
        assert "/comment'" in page
        assert 'id="c-send"' in page

    def test_tables_render(self):
        page = render_artifact("t", "| a | b |\n|---|---|\n| 1 | 2 |\n")
        assert "<table>" in page
