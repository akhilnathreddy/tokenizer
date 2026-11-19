"""Byte-level Byte-Pair Encoding (BPE), the tokenizer family used by GPT-2,
GPT-3/4, LLaMA-3 and many other open LLMs.

How it works
------------
Training
    1. Pre-tokenize the corpus with the GPT-2 regex and count each chunk.
    2. Start with one token per byte (vocab = 256 entries), so *any* text can
       be represented and there is never an unknown token.
    3. Count every adjacent pair of tokens across the corpus (weighted by
       chunk frequency), merge the most frequent pair into a new token, and
       record the merge.  Repeat until the vocabulary reaches ``vocab_size``.

Encoding
    Split the text the same way, turn each chunk into bytes, then repeatedly
    apply the *earliest learned* merge that is present until no learned merge
    applies.  Map the resulting byte strings to ids.

Decoding
    Concatenate the bytes of every token and decode as UTF-8.

Tokens are stored internally as ``bytes``.  For display and for vocab files
they are written with the GPT-2 byte->unicode mapping (a leading space is
shown as ``Ġ``).
"""

from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .base import Tokenizer
from .byte_unicode import bytes_to_repr, repr_to_bytes
from .pretokenize import gpt2_split

ENDOFTEXT = "<|endoftext|>"

Pair = tuple[bytes, bytes]


def display(token: bytes) -> str:
    """Human readable text for a token (invalid UTF-8 shown as \\xNN)."""
    return token.decode("utf-8", errors="backslashreplace")


class ByteLevelBPE(Tokenizer):
    kind = "bpe"

    def __init__(
        self,
        merges: list[Pair],
        vocab: dict[int, bytes] | None = None,
        special_tokens: dict[str, int] | None = None,
    ) -> None:
        self.merges = list(merges)
        self.ranks: dict[Pair, int] = {pair: r for r, pair in enumerate(self.merges)}
        if vocab is None:
            vocab = {b: bytes([b]) for b in range(256)}
            for a, b in self.merges:
                if a + b not in vocab.values():
                    vocab[len(vocab)] = a + b
        self.vocab: dict[int, bytes] = dict(vocab)
        self.token_to_id: dict[bytes, int] = {tok: i for i, tok in self.vocab.items()}
        self.special_tokens: dict[str, int] = dict(special_tokens or {})
        self.special_by_id: dict[int, str] = {i: s for s, i in self.special_tokens.items()}
        self._special_re = (
            re.compile(
                "("
                + "|".join(re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True))
                + ")"
            )
            if self.special_tokens
            else None
        )
        self._cache: dict[bytes, list[bytes]] = {}

    # ================================================================ training
    @classmethod
    def train(
        cls,
        corpus: str | Iterable[str],
        vocab_size: int,
        special_tokens: Iterable[str] = (ENDOFTEXT,),
        min_frequency: int = 2,
        trace: bool = False,
        trace_words: int = 40,
        trace_top_k: int = 10,
    ) -> "ByteLevelBPE":
        """Learn merges from ``corpus`` until the vocab has ``vocab_size`` entries.

        With ``trace=True`` the returned tokenizer has a ``training_trace``
        attribute: a JSON-friendly record of every merge, used by the web
        visualizer to replay training step by step.
        """
        special_tokens = list(special_tokens)
        target = vocab_size - len(special_tokens)
        if target < 256:
            raise ValueError("vocab_size must leave room for the 256 byte tokens")

        # 1. Pre-tokenize and count unique chunks.
        texts = [corpus] if isinstance(corpus, str) else corpus
        splitter = (
            re.compile("|".join(re.escape(s) for s in special_tokens)) if special_tokens else None
        )
        counts: Counter[bytes] = Counter()
        for text in texts:
            for segment in splitter.split(text) if splitter else [text]:
                counts.update(chunk.encode("utf-8") for chunk in gpt2_split(segment))

        chunks = sorted(counts, key=lambda c: (-counts[c], c))
        words: list[list[bytes]] = [[bytes([b]) for b in c] for c in chunks]
        freqs: list[int] = [counts[c] for c in chunks]

        vocab: dict[int, bytes] = {b: bytes([b]) for b in range(256)}
        token_to_id: dict[bytes, int] = {tok: i for i, tok in vocab.items()}
        merges: list[Pair] = []

        # 2. Count pairs and remember which words contain each pair.
        pair_counts: dict[Pair, int] = defaultdict(int)
        where: dict[Pair, set[int]] = defaultdict(set)
        for idx, (seq, freq) in enumerate(zip(words, freqs)):
            for pair in zip(seq, seq[1:]):
                pair_counts[pair] += freq
                where[pair].add(idx)

        # Max-heap with lazy deletion: entries are re-pushed whenever a count
        # changes and stale ones are skipped when popped.  Ties are broken by
        # the smaller byte strings so training is deterministic.
        heap = [(-c, a, b) for (a, b), c in pair_counts.items()]
        heapq.heapify(heap)

        total_tokens = sum(len(s) * f for s, f in zip(words, freqs))
        rec: dict | None = None
        tracked: list[int] = []
        if trace:
            tracked = list(range(min(trace_words, len(words))))
            rec = {
                "corpus": {
                    "characters": sum(len(c.decode("utf-8", "replace")) * counts[c] for c in chunks),
                    "bytes": total_tokens,
                    "tokens": total_tokens,
                    "chunks": sum(freqs),
                    "unique_chunks": len(chunks),
                },
                "words": [
                    {"text": display(chunks[i]), "freq": freqs[i], "parts": [token_to_id[t] for t in words[i]]}
                    for i in tracked
                ],
                "steps": [],
            }
        tracked_pos = {w: k for k, w in enumerate(tracked)}
        stop_reason = "reached vocab_size"

        # 3. Merge loop.
        while len(vocab) < target:
            best: Pair | None = None
            while heap:
                neg, a, b = heapq.heappop(heap)
                if pair_counts.get((a, b), 0) == -neg:
                    best, best_count = (a, b), -neg
                    break
            if best is None:
                stop_reason = "no pairs left to merge"
                break
            if best_count < min_frequency:
                stop_reason = f"no pair occurs at least {min_frequency} times"
                break

            top = None
            if rec is not None:
                top = heapq.nsmallest(trace_top_k, pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))

            a, b = best
            new_tok = a + b
            merges.append(best)
            if new_tok not in token_to_id:
                token_to_id[new_tok] = len(vocab)
                vocab[len(vocab)] = new_tok
            new_id = token_to_id[new_tok]

            changed: set[Pair] = set()
            updates: dict[int, list[int]] = {}
            for idx in list(where[best]):
                seq = words[idx]
                merged: list[bytes] = []
                i = 0
                while i < len(seq):
                    if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                        merged.append(new_tok)
                        i += 2
                    else:
                        merged.append(seq[i])
                        i += 1
                if len(merged) == len(seq):
                    continue  # stale index: the pair is no longer in this word
                freq = freqs[idx]
                for pair in zip(seq, seq[1:]):
                    pair_counts[pair] -= freq
                    changed.add(pair)
                for pair in zip(merged, merged[1:]):
                    pair_counts[pair] += freq
                    where[pair].add(idx)
                    changed.add(pair)
                total_tokens -= (len(seq) - len(merged)) * freq
                words[idx] = merged
                if idx in tracked_pos:
                    updates[tracked_pos[idx]] = [token_to_id[t] for t in merged]

            for pair in changed:
                c = pair_counts[pair]
                if c > 0:
                    heapq.heappush(heap, (-c, pair[0], pair[1]))
                else:
                    pair_counts.pop(pair, None)
                    where.pop(pair, None)
            where.pop(best, None)

            if rec is not None:
                rec["steps"].append(
                    {
                        "rank": len(merges) - 1,
                        "pair": [token_to_id[a], token_to_id[b]],
                        "new_id": new_id,
                        "count": best_count,
                        "top": [[token_to_id[p[0]], token_to_id[p[1]], c] for p, c in top],
                        "updates": updates,
                        "total_tokens": total_tokens,
                    }
                )

        specials = {s: len(vocab) + k for k, s in enumerate(special_tokens)}
        tok = cls(merges, vocab, specials)
        tok.stop_reason = stop_reason
        if rec is not None:
            rec["stop_reason"] = stop_reason
            rec["tokens"] = {i: [display(t), bytes_to_repr(t)] for i, t in vocab.items()}
            tok.training_trace = rec
        return tok

    # ================================================================ encoding
    def _merge_chunk(self, chunk: bytes, steps: list | None = None) -> list[bytes]:
        """Apply learned merges to one pre-token, lowest rank first."""
        if steps is None and chunk in self._cache:
            return self._cache[chunk]
        parts = [bytes([b]) for b in chunk]
        ranks = self.ranks
        while len(parts) > 1:
            best_rank, best = None, None
            for pair in zip(parts, parts[1:]):
                r = ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best = r, pair
            if best is None:
                break
            a, b = best
            merged, i = [], 0
            while i < len(parts):
                if i + 1 < len(parts) and parts[i] == a and parts[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
            if steps is not None:
                steps.append(
                    {
                        "rank": best_rank,
                        "pair": [bytes_to_repr(a), bytes_to_repr(b)],
                        "parts": [bytes_to_repr(p) for p in parts],
                    }
                )
        if steps is None and len(self._cache) < 100_000:
            self._cache[chunk] = parts
        return parts

    def _segments(self, text: str, allowed_special: bool) -> list[tuple[str, bool]]:
        """Split out special tokens: returns (segment, is_special) pairs."""
        if not allowed_special or self._special_re is None:
            return [(text, False)]
        out = []
        for k, piece in enumerate(self._special_re.split(text)):
            if piece:
                out.append((piece, k % 2 == 1))
        return out

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        ids: list[int] = []
        for segment, special in self._segments(text, allowed_special):
            if special:
                ids.append(self.special_tokens[segment])
                continue
            for chunk in gpt2_split(segment):
                ids.extend(self.token_to_id[p] for p in self._merge_chunk(chunk.encode("utf-8")))
        return ids

    def decode_bytes(self, ids: list[int]) -> bytes:
        out = []
        for i in ids:
            if i in self.special_by_id:
                out.append(self.special_by_id[i].encode("utf-8"))
            else:
                out.append(self.vocab[i])
        return b"".join(out)

    def decode(self, ids: list[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def id_to_token(self, idx: int) -> str:
        if idx in self.special_by_id:
            return self.special_by_id[idx]
        return bytes_to_repr(self.vocab[idx])

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    def token_info(self, idx: int) -> dict:
        if idx in self.special_by_id:
            s = self.special_by_id[idx]
            return {"id": idx, "text": s, "repr": s, "bytes": list(s.encode()), "special": True}
        tok = self.vocab[idx]
        return {"id": idx, "text": display(tok), "repr": bytes_to_repr(tok), "bytes": list(tok)}

    def encode_trace(self, text: str) -> dict:
        pretokens = []
        ids: list[int] = []
        for segment, special in self._segments(text, True):
            if special:
                idx = self.special_tokens[segment]
                info = self.token_info(idx)
                pretokens.append({"text": segment, "special": True, "bytes": info["bytes"], "steps": [], "tokens": [info]})
                ids.append(idx)
                continue
            for chunk in gpt2_split(segment):
                raw = chunk.encode("utf-8")
                steps: list = []
                parts = self._merge_chunk(raw, steps)
                toks = [self.token_info(self.token_to_id[p]) for p in parts]
                ids.extend(t["id"] for t in toks)
                pretokens.append(
                    {"text": chunk, "repr": bytes_to_repr(raw), "bytes": list(raw), "steps": steps, "tokens": toks}
                )
        return {
            "kind": self.kind,
            "input": text,
            "normalized": text,
            "pretokens": pretokens,
            "ids": ids,
            "tokens": [t for p in pretokens for t in p["tokens"]],
            "unk_count": 0,
            "vocab_size": self.vocab_size,
            "byte_count": len(text.encode("utf-8")),
        }

    # =========================================================== serialization
    def to_dict(self) -> dict:
        return {
            "merges": [[bytes_to_repr(a), bytes_to_repr(b)] for a, b in self.merges],
            "vocab": {bytes_to_repr(t): i for i, t in self.vocab.items()},
            "special_tokens": self.special_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ByteLevelBPE":
        merges = [(repr_to_bytes(a), repr_to_bytes(b)) for a, b in data["merges"]]
        vocab = {int(i): repr_to_bytes(t) for t, i in data["vocab"].items()}
        return cls(merges, vocab, data.get("special_tokens"))

    # ------------------------------------------------------- GPT-2 file format
    @classmethod
    def from_gpt2(cls, encoder: dict[str, int], merges_text: str) -> "ByteLevelBPE":
        """Build from GPT-2's ``encoder.json`` (parsed) and ``vocab.bpe`` (text)."""
        merges = []
        for line in merges_text.splitlines():
            if not line or line.startswith("#version"):
                continue
            a, b = line.split(" ")
            merges.append((repr_to_bytes(a), repr_to_bytes(b)))
        vocab, specials = {}, {}
        for tok, i in encoder.items():
            if tok.startswith("<|") and tok.endswith("|>"):
                specials[tok] = i
            else:
                vocab[i] = repr_to_bytes(tok)
        return cls(merges, vocab, specials)

    @classmethod
    def from_gpt2_files(cls, encoder_json: str | Path, vocab_bpe: str | Path) -> "ByteLevelBPE":
        encoder = json.loads(Path(encoder_json).read_text(encoding="utf-8"))
        return cls.from_gpt2(encoder, Path(vocab_bpe).read_text(encoding="utf-8"))

    def save_gpt2(self, directory: str | Path) -> None:
        """Write ``encoder.json`` and ``vocab.bpe`` in the GPT-2 format."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        encoder = {bytes_to_repr(t): i for i, t in self.vocab.items()}
        encoder.update(self.special_tokens)
        (d / "encoder.json").write_text(json.dumps(encoder, ensure_ascii=False), encoding="utf-8")
        lines = ["#version: 0.2"] + [f"{bytes_to_repr(a)} {bytes_to_repr(b)}" for a, b in self.merges]
        (d / "vocab.bpe").write_text("\n".join(lines) + "\n", encoding="utf-8")
