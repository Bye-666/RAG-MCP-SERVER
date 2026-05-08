from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base_splitter import BaseSplitter
from src.core.trace import TraceContext


class RecursiveSplitter(BaseSplitter):
    def __init__(
        self,
        provider: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        **kwargs
    ):
        self.provider = provider
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if separators is None:
            separators = [
                "\n## ",
                "\n### ",
                "\n#### ",
                "\n\n",
                "\n",
                " ",
                ""
            ]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False
        )

    def split_text(
        self,
        text: str,
        trace: Optional[TraceContext] = None
    ) -> List[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            return []

        chunks = self.splitter.split_text(text)

        if trace:
            trace.log("splitter", {
                "provider": self.provider,
                "input_length": len(text),
                "chunk_count": len(chunks),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            })

        return chunks
