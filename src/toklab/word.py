"""Word-level tokenizer.

The simplest useful tokenizer: every distinct word (or punctuation mark) in
the training corpus becomes one vocabulary entry.  Anything not seen during
training maps to ``<unk>`` - the out-of-vocabulary problem that motivates
sub-word tokenizers such as BPE and WordPiece.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from .base import Tokenizer
from .normalize import Normalizer
from .pretokenize import split_words

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
DEFAULT_SPECIALS = (PAD, UNK, BOS, EOS)

_NO_SPACE_BEFORE = re.compile(r" ([.,!?;:%)\]}'\"…])")
_NO_SPACE_AFTER = re.compile(r"([(\[{$]) ")


class WordTokenizer(Tokenizer):
    kind = "word"

    def __init__(
        self,
        vocab: dict[str, int],
        normalizer: Normalizer | None = None,
        specials: Iterable[str] = DEFAULT_SPECIALS,
    ) -> None:
        self.normalizer = normalizer or Normalizer(lowercase=True)
        self.specials = list(specials)
        if UNK not in self.specials:
            raise ValueError("word tokenizer needs an <unk> token")
        self.vocab = dict(vocab)
        for tok in self.specials:
            if tok not in self.vocab:
                raise ValueError(f"special token {tok!r} missing from vocab")
        self.inverse = {i: t for t, i in self.vocab.items()}
        self.unk_id = self.vocab[UNK]

    # ------------------------------------------------------------------ train
    @classmethod
    def train(
        cls,
        corpus: str | Iterable[str],
        vocab_size: int | None = None,
        min_freq: int = 1,
        normalizer: Normalizer | None = None,
        specials: Iterable[str] = DEFAULT_SPECIALS,
    ) -> "WordTokenizer":
        """Build a vocabulary from word frequencies.

        Words are sorted by frequency (ties alphabetically) and kept while they
        appear at least ``min_freq`` times and the vocabulary is below
        ``vocab_size`` (special tokens included).
        """
        normalizer = normalizer or Normalizer(lowercase=True)
        specials = list(specials)
        texts = [corpus] if isinstance(corpus, str) else corpus
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(split_words(normalizer(text)))

        vocab = {tok: i for i, tok in enumerate(specials)}
        for word, freq in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq < min_freq or (vocab_size is not None and len(vocab) >= vocab_size):
                break
            if word not in vocab:
                vocab[word] = len(vocab)
        tok = cls(vocab, normalizer, specials)
        tok.word_counts = counts
        return tok

    # ------------------------------------------------------------ encode/decode
    def pretokenize(self, text: str) -> list[str]:
        return split_words(self.normalizer(text))

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.vocab.get(w, self.unk_id) for w in self.pretokenize(text)]
        if add_bos:
            ids.insert(0, self.vocab[BOS])
        if add_eos:
            ids.append(self.vocab[EOS])
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Join words with spaces.

        Word-level tokenization is lossy: the original whitespace is gone, so
        decoding uses a heuristic that removes spaces around punctuation.
        """
        words = []
        for i in ids:
            tok = self.inverse.get(i, UNK)
            if skip_special and tok in self.specials and tok != UNK:
                continue
            words.append(tok)
        text = " ".join(words)
        text = _NO_SPACE_BEFORE.sub(r"\1", text)
        return _NO_SPACE_AFTER.sub(r"\1", text)

    def id_to_token(self, idx: int) -> str:
        return self.inverse.get(idx, UNK)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------ trace
    def encode_trace(self, text: str) -> dict:
        normalized = self.normalizer(text)
        words = split_words(normalized)
        tokens = []
        for w in words:
            idx = self.vocab.get(w, self.unk_id)
            tokens.append({"id": idx, "text": self.inverse[idx], "source": w, "unk": idx == self.unk_id})
        return {
            "kind": self.kind,
            "input": text,
            "normalized": normalized,
            "pretokens": words,
            "tokens": tokens,
            "ids": [t["id"] for t in tokens],
            "unk_count": sum(t["unk"] for t in tokens),
            "vocab_size": self.vocab_size,
        }

    # -------------------------------------------------------------- serialize
    def to_dict(self) -> dict:
        return {
            "normalizer": self.normalizer.to_dict(),
            "specials": self.specials,
            "vocab": self.vocab,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WordTokenizer":
        return cls(
            vocab=data["vocab"],
            normalizer=Normalizer.from_dict(data.get("normalizer")),
            specials=data.get("specials", DEFAULT_SPECIALS),
        )
