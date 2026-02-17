"""Tests for boom.config — configuration and validation."""

import os
import pytest


class TestConfig:
    def test_default_models(self):
        from boom import config

        assert "claude" in config.CLAUDE_MODEL.lower() or "sonnet" in config.CLAUDE_MODEL.lower()
        assert "voyage" in config.VOYAGE_MODEL.lower()
        assert "context" in config.VOYAGE_CONTEXT_MODEL.lower()
        assert "rerank" in config.VOYAGE_RERANK_MODEL.lower()

    def test_default_cors(self):
        from boom import config

        assert isinstance(config.CORS_ORIGINS, list)
        assert len(config.CORS_ORIGINS) >= 1

    def test_default_storage_dir(self):
        from boom import config

        assert config.VECTOR_STORAGE_DIR

    def test_validate_keys_exits_on_missing(self):
        """validate_keys should sys.exit(1) if API keys are empty."""
        from boom import config

        original_anthropic = config.ANTHROPIC_API_KEY
        original_voyage = config.VOYAGE_API_KEY
        try:
            config.ANTHROPIC_API_KEY = ""
            config.VOYAGE_API_KEY = ""
            with pytest.raises(SystemExit) as exc_info:
                config.validate_keys()
            assert exc_info.value.code == 1
        finally:
            config.ANTHROPIC_API_KEY = original_anthropic
            config.VOYAGE_API_KEY = original_voyage

    def test_validate_keys_passes_with_keys(self):
        from boom import config

        original_anthropic = config.ANTHROPIC_API_KEY
        original_voyage = config.VOYAGE_API_KEY
        try:
            config.ANTHROPIC_API_KEY = "test-key"
            config.VOYAGE_API_KEY = "test-key"
            # Should not raise
            config.validate_keys()
        finally:
            config.ANTHROPIC_API_KEY = original_anthropic
            config.VOYAGE_API_KEY = original_voyage
