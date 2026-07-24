"""cli._export_memory — one-way markdown export of the Postgres second
brain into a greppable/diffable file tree (MEMORY.md index + per-fact files)."""
from types import SimpleNamespace

import pytest

from cli import _export_memory

pytestmark = pytest.mark.integration


def _container(memdb, persona_id):
    return SimpleNamespace(
        memory_database=memdb,
        persona=SimpleNamespace(id=persona_id),
    )


class TestExport:
    async def test_writes_index_and_entry_files(self, memdb, persona_id, tmp_path):
        a = await memdb.save_entry(persona_id=persona_id, scope="user",
                                   title="homelab", content="the user runs a homelab")
        await memdb.save_entry(persona_id=persona_id, scope="reference",
                               content="dashboard at https://status.example.com")
        n = await _export_memory(_container(memdb, persona_id), str(tmp_path))
        assert n == 2
        index = (tmp_path / "MEMORY.md").read_text()
        assert "homelab" in index
        assert str(a.id) in index
        entry_file = tmp_path / "entries" / f"{a.id}.md"
        assert entry_file.exists()
        body = entry_file.read_text()
        assert "the user runs a homelab" in body
        assert "scope: user" in body

    async def test_frontmatter_flags_and_links(self, memdb, persona_id, tmp_path):
        a = await memdb.save_entry(persona_id=persona_id, scope="agent",
                                   content="config in src/settings.py", volatile=True)
        b = await memdb.save_entry(persona_id=persona_id, scope="agent",
                                   content="the assistant replies in English")
        await memdb.set_pinned(a.id, True)
        await memdb.add_link(a.id, b.id, "relates_to")
        await _export_memory(_container(memdb, persona_id), str(tmp_path))
        body = (tmp_path / "entries" / f"{a.id}.md").read_text()
        assert "pinned: true" in body
        assert "volatile: true" in body
        assert f"[[{b.id}]]" in body  # wiki-style link to the neighbor

    async def test_core_summaries_exported(self, memdb, persona_id, tmp_path):
        await memdb.set_core(persona_id, "user", "", "the operator narrative", 3)
        await _export_memory(_container(memdb, persona_id), str(tmp_path))
        assert (tmp_path / "core" / "user.md").read_text().find("operator narrative") != -1

    async def test_empty_store_makes_empty_index(self, memdb, persona_id, tmp_path):
        n = await _export_memory(_container(memdb, persona_id), str(tmp_path))
        assert n == 0
        assert (tmp_path / "MEMORY.md").exists()
