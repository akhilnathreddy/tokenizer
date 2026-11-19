"""Common interface shared by every tokenizer in toklab."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path


class Tokenizer(ABC):
    """Abstract tokenizer.

    Every tokenizer can encode text to ids, decode ids back to text, report its
    vocabulary size, serialize itself to a JSON-friendly dict, and produce an
    ``encode_trace`` describing each intermediate stage for the visualizer.
    """

    kind: str = "base"

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str: ...

    @abstractmethod
    def id_to_token(self, idx: int) -> str: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @abstractmethod
    def to_dict(self) -> dict: ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> "Tokenizer": ...

    @abstractmethod
    def encode_trace(self, text: str) -> dict: ...

    def tokenize(self, text: str) -> list[str]:
        """Return the token strings for ``text``."""
        return [self.id_to_token(i) for i in self.encode(text)]

    def save(self, path: str | Path) -> None:
        data = {"kind": self.kind, **self.to_dict()}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load(path: str | Path) -> Tokenizer:
    """Load any toklab tokenizer saved with :meth:`Tokenizer.save`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return from_dict(data)


def from_dict(data: dict) -> Tokenizer:
    from .registry import REGISTRY

    kind = data.get("kind")
    if kind not in REGISTRY:
        raise ValueError(f"unknown tokenizer kind: {kind!r}")
    return REGISTRY[kind].from_dict(data)
