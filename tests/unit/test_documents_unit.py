"""storage.docs chunking + capabilities.documents extraction (no DB)."""
import pytest

from capabilities.documents import extract_text
from storage.docs import CHUNK_CHARS, CHUNK_OVERLAP, chunk_text


class TestChunking:
    def test_short_text_single_chunk(self):
        assert chunk_text("hello world") == ["hello world"]

    def test_empty_is_empty(self):
        assert chunk_text("   ") == []

    def test_long_text_chunks_with_overlap(self):
        text = " ".join(f"word{i}" for i in range(2000))
        chunks = chunk_text(text)
        assert len(chunks) > 1
        assert all(len(c) <= CHUNK_CHARS for c in chunks)
        # Overlap: the start of chunk N+1 repeats the tail of chunk N.
        tail = chunks[0][-50:]
        assert tail.split()[-1] in chunks[1][:CHUNK_OVERLAP + 60]

    def test_prefers_word_boundaries(self):
        text = "supercalifragilistic " * 200
        for c in chunk_text(text):
            assert not c.startswith("upercali"), "chunk should not start mid-word"

    def test_full_text_coverage(self):
        text = "\n".join(f"line {i} with some content" for i in range(500))
        chunks = chunk_text(text)
        # Every line survives somewhere (boundaries may split, so sample).
        for probe in ("line 0 ", "line 250 ", "line 499 "):
            assert any(probe in c for c in chunks)


class TestExtractText:
    def test_plain_text(self):
        assert extract_text("text/plain", "hola\namigo".encode()) == "hola\namigo"

    def test_markdown_is_text(self):
        assert "# title" in extract_text("text/markdown", b"# title")

    def test_images_not_ingestible(self):
        assert extract_text("image/jpeg", b"\xff\xd8\xff") is None

    def test_unknown_binary_not_ingestible(self):
        assert extract_text("application/zip", b"PK\x03\x04") is None

    def test_pdf_extracts(self):
        from pypdf import PdfWriter
        import io
        w = PdfWriter()
        w.add_blank_page(width=200, height=200)
        buf = io.BytesIO()
        w.write(buf)
        # A blank page has no text — extraction succeeds with empty string.
        out = extract_text("application/pdf", buf.getvalue())
        assert out is not None

    def test_corrupt_pdf_returns_none_or_empty(self):
        out = extract_text("application/pdf", b"not a pdf at all")
        assert not out
