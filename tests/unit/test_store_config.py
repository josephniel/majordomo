"""Retrieval settings must actually reach the stores that use them.

`EMBEDDING_MODEL` and every `RERANK_*` knob were documented, parsed from
os.environ — and inert. The adapters read them into module constants at
IMPORT time, and the composition root imports `adapters.store` before it
loads the instance config, so the values were frozen from the ambient shell
before the config existed. Setting them changed nothing and said nothing.

Reading configuration at import time is reading it before it exists. These
tests pin the fix: config is parsed in one place and handed to the objects
that need it, so "did my setting take effect" is answerable by looking at
the object.
"""
import pytest

from adapters.store import DocumentStore, Embedder, MemoryDatabase, RerankConfig, Reranker
from runtime.settings import RuntimeSettings


class TestSettingsCarryTheRetrievalConfig:
    def test_the_embedding_model_is_parsed(self):
        s = RuntimeSettings.from_env({"EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5"})
        assert s.embedding_model == "BAAI/bge-base-en-v1.5"

    def test_every_rerank_knob_is_parsed(self):
        """All five were documented in .env.example and all five did
        nothing."""
        s = RuntimeSettings.from_env({
            "RERANK_ENABLED": "0",
            "RERANK_MODEL": "some/other-cross-encoder",
            "RERANK_CANDIDATES": "7",
            "RERANK_CENTER": "-3.5",
            "RERANK_TEMPERATURE": "1.25",
        })
        assert s.rerank == RerankConfig(
            enabled=False, model="some/other-cross-encoder",
            candidates=7, center=-3.5, temperature=1.25,
        )

    def test_an_empty_environment_gets_the_measured_defaults(self):
        s = RuntimeSettings.from_env({})
        assert s.rerank == RerankConfig()
        assert s.embedding_model == ""

    def test_rerank_enabled_defaults_on_not_off(self):
        """`enabled` is the one boolean where the default matters: parsing
        an absent variable as False would silently disable reranking for
        every operator who never set it."""
        assert RuntimeSettings.from_env({}).rerank.enabled is True
        assert RuntimeSettings.from_env({"RERANK_ENABLED": ""}).rerank.enabled is True
        assert RuntimeSettings.from_env({"RERANK_ENABLED": "0"}).rerank.enabled is False


class TestTheStoresUseWhatTheyAreGiven:
    def test_the_memory_store_embeds_with_the_injected_model(self):
        e = Embedder("BAAI/bge-base-en-v1.5")
        db = MemoryDatabase("postgres://x/y", embedder=e)
        assert db.embedder.model_name == "BAAI/bge-base-en-v1.5"

    def test_the_vector_width_follows_the_model(self):
        """The dimension is not a constant — it sizes the vector column, and
        getting it from the wrong model is a silent insert failure."""
        assert Embedder("BAAI/bge-base-en-v1.5").dim == 768
        assert Embedder("mixedbread-ai/mxbai-embed-large-v1").dim == 1024

    def test_an_unknown_model_is_refused_by_name(self):
        with pytest.raises(ValueError, match="not a supported fastembed model"):
            _ = Embedder("definitely/not-a-real-model-xyz").dim

    def test_the_document_store_takes_one_too(self):
        """Both tables live in one database; if the two stores disagreed
        about the model, one would resize the other's column."""
        e = Embedder("BAAI/bge-base-en-v1.5")
        assert DocumentStore("postgres://x/y", embedder=e)._embed is e

    def test_reranking_can_be_turned_off(self):
        assert Reranker(RerankConfig(enabled=False)).available() is False
        assert Reranker(RerankConfig(enabled=True)).available() is True

    def test_the_candidate_depth_is_the_configured_one(self):
        assert Reranker(RerankConfig(candidates=7)).candidates == 7

    def test_calibration_follows_the_configured_curve(self):
        """center/temperature define where the decision boundary sits. If
        they were ignored, a tuned threshold would silently mean something
        different from what was measured."""
        logit = -8.0
        assert Reranker(RerankConfig(center=-8.0)).\
            _calibrate(logit) == pytest.approx(0.5)
        # Move the center and the same logit must no longer read as neutral.
        assert Reranker(RerankConfig(center=-4.0))._calibrate(logit) < 0.2


class TestPersonasSharingADatabaseMustAgree:
    """The hazard that fixing the bug above opens.

    While EMBEDDING_MODEL was inert, every persona silently used the default
    and this could not happen. Honouring it means two personas pointed at one
    database can now ask for different models — and vector width is a
    property of the TABLE: `init_schema` resizes memory_entries.embedding and
    CLEARS every vector to do it. The second persona to start would wipe the
    first one's semantic index, and nothing would report it, because recall
    degrades to FTS + trigram and keeps answering.
    """

    def _project(self, tmp_path, personas):
        for pid, env in personas.items():
            d = tmp_path / "instances" / pid
            d.mkdir(parents=True)
            (d / "persona.yaml").write_text(f"name: {pid}\n")
            (d / ".env").write_text("".join(f"{k}={v}\n" for k, v in env.items()))
        return tmp_path

    def _runtime(self, tmp_path, pid, env):
        from runtime.container import PersonaRuntime
        from runtime.persona import Persona
        rt = PersonaRuntime(Persona(
            id=pid, dir=tmp_path / "instances" / pid, name=pid, system_prompt="",
        ))
        rt.settings = RuntimeSettings.from_env(env)  # bypass the ambient shell
        return rt

    def test_a_conflict_on_one_database_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        dsn = "postgres://majordomo:x@127.0.0.1:5433/shared"
        root = self._project(tmp_path, {
            "alice": {"MEMORY_DATABASE_URL": dsn, "EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5"},
            "bob": {"MEMORY_DATABASE_URL": dsn, "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5"},
        })
        rt = self._runtime(root, "alice", {
            "MEMORY_DATABASE_URL": dsn, "EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5",
        })
        with pytest.raises(SystemExit) as exc:
            rt._assert_embedding_model_is_host_wide(dsn)
        msg = str(exc.value)
        assert "alice" in msg
        assert "bob" in msg
        assert "bge-base-en-v1.5" in msg
        assert "bge-small-en-v1.5" in msg

    def test_the_refusal_does_not_leak_the_password(self, tmp_path, monkeypatch):
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        dsn = "postgres://majordomo:hunter2@127.0.0.1:5433/shared"
        root = self._project(tmp_path, {
            "alice": {"MEMORY_DATABASE_URL": dsn, "EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5"},
            "bob": {"MEMORY_DATABASE_URL": dsn, "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5"},
        })
        rt = self._runtime(root, "alice", {
            "MEMORY_DATABASE_URL": dsn, "EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5",
        })
        with pytest.raises(SystemExit) as exc:
            rt._assert_embedding_model_is_host_wide(dsn)
        assert "hunter2" not in str(exc.value)

    def test_separate_databases_may_use_different_models(self, tmp_path, monkeypatch):
        """Personas don't have to share a database, and two that don't are
        free to disagree — so the check is on the DSN, not a blanket ban."""
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        mine = "postgres://majordomo:x@127.0.0.1:5433/alice_db"
        root = self._project(tmp_path, {
            "alice": {"MEMORY_DATABASE_URL": mine, "EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5"},
            "bob": {"MEMORY_DATABASE_URL": "postgres://majordomo:x@127.0.0.1:5433/bob_db",
                    "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5"},
        })
        rt = self._runtime(root, "alice", {
            "MEMORY_DATABASE_URL": mine, "EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5",
        })
        rt._assert_embedding_model_is_host_wide(mine)  # must not raise

    def test_agreeing_by_both_saying_nothing_is_agreement(self, tmp_path, monkeypatch):
        """The overwhelmingly common case: neither persona configures a
        model, both get the default. An implementation comparing raw strings
        without resolving the default would compare '' to '' and pass here
        but fail the next test."""
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        dsn = "postgres://majordomo:x@127.0.0.1:5433/shared"
        root = self._project(tmp_path, {
            "alice": {"MEMORY_DATABASE_URL": dsn},
            "bob": {"MEMORY_DATABASE_URL": dsn},
        })
        rt = self._runtime(root, "alice", {"MEMORY_DATABASE_URL": dsn})
        rt._assert_embedding_model_is_host_wide(dsn)  # must not raise

    def test_unset_and_explicitly_the_default_are_the_same_model(self, tmp_path, monkeypatch):
        """One persona spells out the default, the other leaves it blank.
        Comparing the configured strings would call that a conflict and
        refuse to start a perfectly valid pair."""
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        dsn = "postgres://majordomo:x@127.0.0.1:5433/shared"
        root = self._project(tmp_path, {
            "alice": {"MEMORY_DATABASE_URL": dsn},
            "bob": {"MEMORY_DATABASE_URL": dsn, "EMBEDDING_MODEL": Embedder().model_name},
        })
        rt = self._runtime(root, "alice", {"MEMORY_DATABASE_URL": dsn})
        rt._assert_embedding_model_is_host_wide(dsn)  # must not raise

    def test_a_sibling_with_an_unreadable_env_does_not_block_startup(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        dsn = "postgres://majordomo:x@127.0.0.1:5433/shared"
        root = self._project(tmp_path, {"alice": {"MEMORY_DATABASE_URL": dsn}})
        broken = root / "instances" / "broken"
        broken.mkdir(parents=True)
        (broken / "persona.yaml").write_text("name: broken\n")  # no .env at all
        rt = self._runtime(root, "alice", {"MEMORY_DATABASE_URL": dsn})
        rt._assert_embedding_model_is_host_wide(dsn)  # must not raise


class TestNoImportTimeState:
    def test_importing_the_adapter_does_not_read_the_environment(self, monkeypatch):
        """The original bug in one assertion: set the variable, import the
        module, and observe that the module never had a chance to see it —
        because nothing about the module depends on the environment at all.
        """
        monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        import importlib

        from adapters.store import embeddings
        importlib.reload(embeddings)
        try:
            # A fresh Embedder takes the documented default, NOT the env var:
            # the environment is the composition root's business, and it is
            # read there and passed in.
            assert embeddings.Embedder().model_name == embeddings.DEFAULT_MODEL
        finally:
            importlib.reload(embeddings)

    def test_two_embedders_do_not_share_mutable_state(self):
        """Model choice used to be a module global, so 'configuring' one
        store reconfigured every other store in the process."""
        a, b = Embedder("BAAI/bge-base-en-v1.5"), Embedder("BAAI/bge-small-en-v1.5")
        assert (a.dim, b.dim) == (768, 384)
        assert a.model_name != b.model_name
