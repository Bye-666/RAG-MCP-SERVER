import pytest
from src.core.settings import Settings, load_settings
from pathlib import Path
import yaml


def test_settings_loading(tmp_path):
    # Create valid settings file
    settings_path = tmp_path / "settings.yaml"
    settings_content = {
        "llm": {"provider": "openai"},
        "embedding": {"provider": "openai"},
        "vector_store": {"provider": "chroma"},
        "retrieval": {"top_k": 5},
        "observability": {"log_level": "INFO"}
    }
    with open(settings_path, "w", encoding="utf-8") as f:
        yaml.dump(settings_content, f)

    settings = load_settings(str(settings_path))
    assert isinstance(settings, Settings)
    assert settings.llm["provider"] == "openai"
    assert settings.retrieval["top_k"] == 5


def test_missing_section(tmp_path):
    # Create settings missing a required section
    settings_path = tmp_path / "settings.yaml"
    settings_content = {
        "llm": {"provider": "openai"},
        # Missing embedding section
        "vector_store": {"provider": "chroma"}
    }
    with open(settings_path, "w", encoding="utf-8") as f:
        yaml.dump(settings_content, f)

    with pytest.raises(ValueError) as exc_info:
        load_settings(str(settings_path))
    assert "Missing required configuration section: embedding" in str(exc_info.value)