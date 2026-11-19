"""WordPiece tokenizer (the BERT family).

Training
    Start from single characters.  The first character of a word is stored
    as-is; characters that continue a word get a ``##`` prefix, so "cat" is
    ``c ##a ##t``.  At every step, score each adjacent pair as::

        score(a, b) = count(ab) / (count(a) * count(b))

    and merge the best-scoring pair.  Unlike BPE, which picks the most
    *frequent* pair, WordPiece prefers pairs whose parts rarely appear apart,
    i.e. merges that most increase the likelihood of the training data.

Encoding
    For each word, repeatedly take the *longest* vocabulary entry matching
    the start of what is left (greedy longest-match-first).  If some part of
    the word cannot be matched, the whole word becomes ``[UNK]``.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable

from .base import Tokenizer
from .normalize import Normalizer
from .pretokenize import split_words

PAD, UNK, CLS, SEP, MASK = "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
DEFAULT_SPECIALS = (PAD, UNK, CLS, SEP, MASK)
PREFIX = "##"

_NO_SPACE_BEFORE = re.compile(r" ([.,!?;:%)\]}'\"…])")


def _join(a: str, b: str) -> str:
    return a + (b[len(PREFIX):] if b.startswith(PREFIX) else b)


class WordPiece(Tokenizer):
    kind = "wordpiece"

    def __init__(
        self,
        vocab: dict[str, int],
        normalizer: Normalizer | None = None,
        specials: Iterable[str] = DEFAULT_SPECIALS,
        max_chars_per_word: int = 100,
    ) -> None:
        self.vocab = dict(vocab)
        self.inverse = {i: t for t, i in self.vocab.items()}
        self.normalizer = normalizer or Normalizer(form="NFC", lowercase=True, strip_accents=True)
        self.specials = list(specials)
        self.unk_id = self.vocab[UNK]
        self.max_chars_per_word = max_chars_per_word

    # ================================================================ training
    @classmethod
    def train(
        cls,
        corpus: str | Iterable[str],
        vocab_size: int,
        normalizer: Normalizer | None = None,
        specials: Iterable[str] = DEFAULT_SPECIALS,
        trace: bool = False,
        trace_words: int = 40,
        trace_top_k: int = 10,
    ) -> "WordPiece":
        normalizer = normalizer or Normalizer(form="NFC", lowercase=True, strip_accents=True)
        specials = list(specials)
        texts = [corpus] if isinstance(corpus, str) else corpus
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(split_words(normalizer(text)))

        words = sorted(counts, key=lambda w: (-counts[w], w))
        freqs = [counts[w] for w in words]
        splits = [[w[0]] + [PREFIX + c for c in w[1:]] for w in words]

        vocab: dict[str, int] = {s: i for i, s in enumerate(specials)}
        for unit in sorted({u for s in splits for u in s}):
            vocab[unit] = len(vocab)

        rec = None
        tracked = list(range(min(trace_words, len(words))))
        if trace:
            rec = {
                "corpus": {"words": sum(freqs), "unique_words": len(words), "alphabet": len(vocab) - len(specials)},
                "words": [{"text": words[i], "freq": freqs[i], "parts": [vocab[u] for u in splits[i]]} for i in tracked],
                "steps": [],
            }

        stop_reason = "reached vocab_size"
        merges = 0
        while len(vocab) < vocab_size:
            unit_counts: dict[str, int] = defaultdict(int)
            pair_counts: dict[tuple[str, str], int] = defaultdict(int)
            for s, f in zip(splits, freqs):
                for u in s:
                    unit_counts[u] += f
                for pair in zip(s, s[1:]):
                    pair_counts[pair] += f
            if not pair_counts:
                stop_reason = "no pairs left to merge"
                break

            def score(item):
                (a, b), c = item
                return c / (unit_counts[a] * unit_counts[b])

            ranked = sorted(pair_counts.items(), key=lambda kv: (-score(kv), -kv[1], kv[0]))
            (a, b), best_count = ranked[0]
            best_score = score(ranked[0])
            new = _join(a, b)
            if new not in vocab:
                vocab[new] = len(vocab)

            updates = {}
            for idx, s in enumerate(splits):
                if len(s) < 2:
                    continue
                merged, i, changed = [], 0, False
                while i < len(s):
                    if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                        merged.append(new)
                        i += 2
                        changed = True
                    else:
                        merged.append(s[i])
                        i += 1
                if changed:
                    splits[idx] = merged
                    if rec is not None and idx < len(tracked):
                        updates[idx] = [vocab[u] for u in merged]
            merges += 1

            if rec is not None:
                rec["steps"].append(
                    {
                        "rank": merges - 1,
                        "pair": [vocab[a], vocab[b]],
                        "new_id": vocab[new],
                        "count": best_count,
                        "score": best_score,
                        "top": [[vocab[p], vocab[q], c, score(((p, q), c))] for (p, q), c in ranked[:trace_top_k]],
                        "updates": updates,
                        "total_tokens": sum(len(s) * f for s, f in zip(splits, freqs)),
                    }
                )

        tok = cls(vocab, normalizer, specials)
        tok.stop_reason = stop_reason
        if rec is not None:
            rec["stop_reason"] = stop_reason
            rec["tokens"] = {i: [t, t] for t, i in vocab.items()}
            tok.training_trace = rec
        return tok

    # ================================================================ encoding
    def _encode_word(self, word: str, steps: list | None = None) -> list[str]:
        if len(word) > self.max_chars_per_word:
            return [UNK]
        pieces: list[str] = []
        start = 0
        while start < len(word):
            end = len(word)
            tried = []
            match = None
            while start < end:
                cand = word[start:end]
                if start > 0:
                    cand = PREFIX + cand
                tried.append(cand)
                if cand in self.vocab:
                    match = cand
                    break
                end -= 1
            if steps is not None:
                steps.append({"start": start, "tried": tried, "match": match})
            if match is None:
                return [UNK]
            pieces.append(match)
            start = end
        return pieces

    def pretokenize(self, text: str) -> list[str]:
        return split_words(self.normalizer(text))

    def encode(self, text: str, add_cls_sep: bool = False) -> list[int]:
        ids = [self.vocab[p] for w in self.pretokenize(text) for p in self._encode_word(w)]
        if add_cls_sep:
            ids = [self.vocab[CLS]] + ids + [self.vocab[SEP]]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        out: list[str] = []
        for i in ids:
            tok = self.inverse.get(i, UNK)
            if skip_special and tok in self.specials and tok != UNK:
                continue
            if tok.startswith(PREFIX) and out:
                out[-1] += tok[len(PREFIX):]
            else:
                out.append(tok)
        return _NO_SPACE_BEFORE.sub(r"\1", " ".join(out))

    def id_to_token(self, idx: int) -> str:
        return self.inverse.get(idx, UNK)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode_trace(self, text: str) -> dict:
        normalized = self.normalizer(text)
        pretokens = []
        for w in split_words(normalized):
            steps: list = []
            pieces = self._encode_word(w, steps)
            toks = [{"id": self.vocab[p], "text": p, "repr": p, "unk": p == UNK} for p in pieces]
            pretokens.append({"text": w, "steps": steps, "tokens": toks})
        tokens = [t for p in pretokens for t in p["tokens"]]
        return {
            "kind": self.kind,
            "input": text,
            "normalized": normalized,
            "pretokens": pretokens,
            "tokens": tokens,
            "ids": [t["id"] for t in tokens],
            "unk_count": sum(t["unk"] for t in tokens),
            "vocab_size": self.vocab_size,
        }

    # =========================================================== serialization
    def to_dict(self) -> dict:
        return {
            "normalizer": self.normalizer.to_dict(),
            "specials": self.specials,
            "max_chars_per_word": self.max_chars_per_word,
            "vocab": self.vocab,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WordPiece":
        return cls(
            data["vocab"],
            Normalizer.from_dict(data.get("normalizer")),
            data.get("specials", DEFAULT_SPECIALS),
            data.get("max_chars_per_word", 100),
        )
