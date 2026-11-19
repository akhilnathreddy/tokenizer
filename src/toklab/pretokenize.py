"""Pre-tokenization: split normalized text into chunks before the model runs.

Two splitters live here:

* ``split_words``   - a simple word/punctuation splitter used by the word-level
                      and WordPiece tokenizers (BERT style).
* ``gpt2_split``    - an exact, dependency-free re-implementation of the GPT-2
                      pre-tokenization regex::

                          's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+

  Python's built-in ``re`` module has no ``\\p{L}`` / ``\\p{N}`` classes, so
  instead of pulling in the third-party ``regex`` package we scan the string
  by hand and classify characters with ``unicodedata``.  The scanner tries the
  same alternatives, in the same order, as the regex engine would.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

GPT2_PATTERN = (
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# Characters with the Unicode White_Space property (what \s means in the regex).
_WHITESPACE = frozenset(
    "\t\n\x0b\x0c\r \x85\xa0 "
    "           "
    "    　"
)

LETTER, NUMBER, SPACE, OTHER = "L", "N", "S", "O"


@lru_cache(maxsize=1 << 16)
def char_class(ch: str) -> str:
    """Classify one character as Letter, Number, Space or Other."""
    if ch in _WHITESPACE:
        return SPACE
    cat = unicodedata.category(ch)
    if cat[0] == "L":
        return LETTER
    if cat[0] == "N":
        return NUMBER
    return OTHER


def gpt2_split(text: str) -> list[str]:
    """Split ``text`` exactly like GPT-2's pre-tokenization regex."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]

        # 1. English contractions: 's 't 're 've 'm 'll 'd
        if ch == "'" and i + 1 < n:
            if text[i + 1 : i + 3] in ("re", "ve", "ll"):
                out.append(text[i : i + 3])
                i += 3
                continue
            if text[i + 1] in "stmd":
                out.append(text[i : i + 2])
                i += 2
                continue

        # 2-4. An optional single space followed by a run of letters,
        #      numbers, or "other" symbols.
        start = i + 1 if ch == " " and i + 1 < n and char_class(text[i + 1]) != SPACE else i
        kind = char_class(text[start])
        if kind != SPACE:
            end = start + 1
            while end < n and char_class(text[end]) == kind:
                end += 1
            out.append(text[i:end])
            i = end
            continue

        # 5-6. Whitespace.  \s+(?!\S) keeps the last space of a run for the
        #      next word, so " hello" stays one chunk.
        end = i + 1
        while end < n and char_class(text[end]) == SPACE:
            end += 1
        if end == n or end - i == 1:
            out.append(text[i:end])
            i = end
        else:
            out.append(text[i : end - 1])
            i = end - 1
    return out


_WORD_RE = re.compile(r"\w+|[^\w\s]")


def split_words(text: str) -> list[str]:
    """Split into runs of word characters and single punctuation marks."""
    return _WORD_RE.findall(text)
