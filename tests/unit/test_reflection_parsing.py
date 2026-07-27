"""capabilities.reflection — defensive JSON fact parsing."""
from domain.reflection import _looks_volatile, _parse_facts


class TestParseFacts:
    def test_clean_json_array(self):
        facts = _parse_facts('[{"scope":"user","content":"x","domain_key":"","title":"t"}]')
        assert len(facts) == 1
        assert facts[0]["content"] == "x"

    def test_empty_array(self):
        assert _parse_facts("[]") == []

    def test_code_fenced(self):
        assert _parse_facts('```json\n[{"scope":"user","content":"y"}]\n```')

    def test_bare_fence(self):
        assert _parse_facts('```\n[{"scope":"user","content":"y"}]\n```')

    def test_prose_wrapped(self):
        raw = 'Here are the facts: [{"scope":"user","content":"z"}] — hope that helps!'
        assert _parse_facts(raw)[0]["content"] == "z"

    def test_no_json_returns_empty(self):
        assert _parse_facts("I could not find any durable facts.") == []

    def test_invalid_json_returns_empty(self):
        assert _parse_facts('[{"scope": "user", broken}]') == []

    def test_non_list_json_returns_empty(self):
        assert _parse_facts('{"scope":"user","content":"x"}') == []

    def test_empty_and_none_input(self):
        assert _parse_facts("") == []
        assert _parse_facts(None) == []

    def test_items_missing_content_dropped(self):
        facts = _parse_facts('[{"scope":"user"},{"scope":"user","content":"keep"}]')
        assert len(facts) == 1
        assert facts[0]["content"] == "keep"

    def test_items_missing_scope_dropped(self):
        assert _parse_facts('[{"content":"no scope"}]') == []

    def test_non_dict_items_dropped(self):
        facts = _parse_facts('["just a string", {"scope":"user","content":"ok"}]')
        assert len(facts) == 1

    def test_multiple_facts_preserved_in_order(self):
        raw = ('[{"scope":"user","content":"first"},'
               '{"scope":"domain","domain_key":"gmail","content":"second"}]')
        facts = _parse_facts(raw)
        assert [f["content"] for f in facts] == ["first", "second"]


class TestLooksVolatile:
    def test_file_path(self):
        assert _looks_volatile("config lives at src/personas/settings.py")

    def test_cli_flag(self):
        assert _looks_volatile("the deploy command takes --prod")

    def test_version(self):
        assert _looks_volatile("the service pins postgres 5.2")

    def test_env_var(self):
        assert _looks_volatile("PRIMARY_LLM selects the chat vendor")

    def test_plain_fact_not_volatile(self):
        assert not _looks_volatile("the user enjoys hiking on weekends")

    def test_name_not_volatile(self):
        assert not _looks_volatile("the user lives in Makati and likes mango")

    def test_empty_not_volatile(self):
        assert not _looks_volatile("")
