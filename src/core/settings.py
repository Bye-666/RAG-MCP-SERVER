from dataclasses import dataclass
import yaml
from pathlib import Path


@dataclass
class Settings:
    llm: dict
    embedding: dict
    vector_store: dict
    retrieval: dict
    observability: dict
    splitter: dict = None  # Optional splitter configuration

    def __post_init__(self):
        # Provide default splitter config if not specified
        if self.splitter is None:
            self.splitter = {
                "splitter": {
                    "provider": "recursive",
                    "chunk_size": 500,
                    "chunk_overlap": 50
                }
            }


def load_settings(path: str = "config/settings.yaml") -> Settings:
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Handle config wrapper if present
    if "config" in config:
        config = config["config"]

    # Validate required sections
    required_sections = ["llm", "embedding", "vector_store", "retrieval", "observability"]
    # Validate vector_store provider configuration
    if config['vector_store']['provider'] == 'qdrant':
        required_keys = ['host', 'port', 'collection_name', 'vector_size']
        for key in required_keys:
            if key not in config['vector_store']:
                raise ValueError(f"Missing {key} in vector_store configuration for Qdrant provider")

    return Settings(**config)
