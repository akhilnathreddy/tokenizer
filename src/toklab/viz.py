"""JSON API used by the web visualizer.

The web page runs this package inside the browser with Pyodide (CPython
compiled to WebAssembly) and calls these functions from JavaScript.  Every
function returns a JSON string, so the page only renders - all tokenization
logic is the same Python code that the tests check.
"""

from __future__ import annotations

import json

from .base import Tokenizer
from .bpe import ByteLevelBPE
from .word import WordTokenizer
from .wordpiece import WordPiece

DEFAULT_SIZES = {"word": 5000, "bpe": 600, "wordpiece": 600}

_models: dict[str, Tokenizer] = {}
_corpus = ""
_gpt2_files: tuple[str, str] | None = None


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def setup(corpus: str, gpt2_encoder: str | None = None, gpt2_merges: str | None = None) -> str:
    """Train the default word, WordPiece and BPE models on ``corpus``."""
    global _corpus, _gpt2_files
    _corpus = corpus
    if gpt2_encoder and gpt2_merges:
        _gpt2_files = (gpt2_encoder, gpt2_merges)
    _models["word"] = WordTokenizer.train(corpus, vocab_size=DEFAULT_SIZES["word"])
    _models["bpe"] = ByteLevelBPE.train(corpus, vocab_size=DEFAULT_SIZES["bpe"])
    _models["wordpiece"] = WordPiece.train(corpus, vocab_size=DEFAULT_SIZES["wordpiece"])
    return info()


def load_gpt2(encoder_json: str, merges_text: str) -> str:
    _models["gpt2"] = ByteLevelBPE.from_gpt2(json.loads(encoder_json), merges_text)
    return info()


def info() -> str:
    return _dump(
        {
            kind: {"vocab_size": tok.vocab_size, "stop_reason": getattr(tok, "stop_reason", "")}
            for kind, tok in _models.items()
        }
    )


def train(kind: str, corpus: str, vocab_size: int) -> str:
    """Train a BPE or WordPiece model with a full step-by-step trace.

    The trained model replaces the current one of that kind, so the Encode
    and Compare views use it straight away.
    """
    vocab_size = int(vocab_size)
    if kind == "bpe":
        tok = ByteLevelBPE.train(corpus, vocab_size=vocab_size, trace=True)
    elif kind == "wordpiece":
        tok = WordPiece.train(corpus, vocab_size=vocab_size, trace=True)
    elif kind == "word":
        tok = WordTokenizer.train(corpus, vocab_size=vocab_size)
        _models[kind] = tok
        return _dump({"kind": kind, "vocab_size": tok.vocab_size})
    else:
        raise ValueError(kind)
    _models[kind] = tok
    trace = tok.training_trace
    trace["kind"] = kind
    trace["vocab_size"] = tok.vocab_size
    return _dump(trace)


def encode(kind: str, text: str) -> str:
    return _dump(_models[kind].encode_trace(text))


def compare(text: str) -> str:
    out = []
    for kind in ("word", "wordpiece", "bpe", "gpt2"):
        if kind in _models:
            trace = _models[kind].encode_trace(text)
            trace.pop("pretokens", None)
            trace["model"] = kind
            out.append(trace)
    return _dump(out)
