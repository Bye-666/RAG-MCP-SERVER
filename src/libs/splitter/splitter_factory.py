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
            raise ValueError(f"Unsupported Splitter provider: {provider}")

        return splitter_class(**settings['splitter'])