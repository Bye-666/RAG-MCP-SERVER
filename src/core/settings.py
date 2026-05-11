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
    splitter: dict = None  # 可选的分割器配置
    storage: dict = None  # 可选的存储配置

    def __post_init__(self):
        # 如果未指定，提供默认的分割器配置
        if self.splitter is None:
            self.splitter = {
                "splitter": {
                    "provider": "recursive",
                    "chunk_size": 500,
                    "chunk_overlap": 50
                }
            }

        # 如果未指定，提供默认的存储配置
        if self.storage is None:
            self.storage = {
                "upload_directory": "./data/uploads"
            }


def load_settings(path: str = "config/settings.yaml") -> Settings:
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # 如果存在配置包装器，则处理
    if "config" in config:
        config = config["config"]

    # 验证必需的配置节
    required_sections = ["llm", "embedding", "vector_store", "retrieval", "observability"]
    # 验证 vector_store 提供者配置
    if config['vector_store']['provider'] == 'qdrant':
        required_keys = ['host', 'port', 'collection_name', 'vector_size']
        for key in required_keys:
            if key not in config['vector_store']:
                raise ValueError(f"Qdrant 提供者的 vector_store 配置中缺少 {key}")

    return Settings(**config)
