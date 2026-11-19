"""Text normalization: the first stage of every tokenizer pipeline."""

from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass


@dataclass
class Normalizer:
    """Configurable text normalizer.

    form:          Unicode normalization form ("NFC", "NFKC", "NFD", "NFKD") or None.
    lowercase:     fold text to lower case.
    strip_accents: remove combining marks (e.g. "café" -> "cafe").
    """

    form: str | None = "NFC"
    lowercase: bool = False
    strip_accents: bool = False

    def __call__(self, text: str) -> str:
        if self.strip_accents:
            text = unicodedata.normalize("NFD", text)
            text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        if self.form:
            text = unicodedata.normalize(self.form, text)
        if self.lowercase:
            text = text.lower()
        return text

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "Normalizer":
        return cls(**(data or {}))
