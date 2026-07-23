"""platforms.config — platform-keyed platform.yaml parsing."""
import pytest

from platforms.config import PlatformConfig


def _write(tmp_path, content):
    (tmp_path / "platform.yaml").write_text(content, encoding="utf-8")
    return tmp_path


class TestPlatformConfig:
    def test_single_block_selected(self, tmp_path):
        _write(tmp_path, "telegram:\n  allowed_user_ids: [7]\n")
        cfg = PlatformConfig.load(tmp_path)
        assert cfg.type == "telegram"
        assert cfg.raw == {"allowed_user_ids": [7]}

    def test_other_platform_block(self, tmp_path):
        _write(tmp_path, "discord:\n  guild_id: 42\n")
        cfg = PlatformConfig.load(tmp_path)
        assert cfg.type == "discord"
        assert cfg.raw == {"guild_id": 42}

    def test_empty_blocks_ignored(self, tmp_path):
        # A template can stub other platforms with null/empty blocks.
        _write(tmp_path, "discord:\ntelegram:\n  allowed_user_ids: [7]\n")
        cfg = PlatformConfig.load(tmp_path)
        assert cfg.type == "telegram"

    def test_no_block_rejected(self, tmp_path):
        _write(tmp_path, "discord:\n")
        with pytest.raises(ValueError, match="no platform block"):
            PlatformConfig.load(tmp_path)

    def test_multiple_blocks_rejected(self, tmp_path):
        _write(tmp_path, "telegram:\n  a: 1\ndiscord:\n  b: 2\n")
        with pytest.raises(ValueError, match="multiple platform blocks"):
            PlatformConfig.load(tmp_path)

    def test_legacy_type_shape_rejected_with_migration_hint(self, tmp_path):
        _write(tmp_path, "type: telegram\nallowed_user_ids: [7]\n")
        with pytest.raises(ValueError, match="no longer supported"):
            PlatformConfig.load(tmp_path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PlatformConfig.load(tmp_path)
