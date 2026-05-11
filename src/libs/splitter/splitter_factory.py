from .base_splitter import BaseSplitter
from typing import Dict, Type


class SplitterFactory:
    _providers: Dict[str, Type[BaseSplitter]] = {}

    @classmethod
    def register_provider(cls, name: str, splitter_class: Type[BaseSplitter]):
        cls._providers[name] = splitter_class

    @classmethod
    def create(cls, settings: dict) -> BaseSplitter:
        provider = settings['splitter']['provider']
        splitter_class = cls._providers.get(provider)

        if not splitter_class:
            raise ValueError(f"不支持的分割器提供商: {provider}")

        return splitter_class(**settings['splitter'])


# Register built-in providers
try:
    from .recursive_splitter import RecursiveSplitter
    SplitterFactory.register_provider("recursive", RecursiveSplitter)
except ImportError:
    pass  # langchain-text-splitters 未安装