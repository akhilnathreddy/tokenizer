"""Command line interface.

    toklab train --model bpe --vocab-size 1000 data/corpus.txt -o bpe.json
    toklab encode -m bpe.json "Hello world"
    toklab encode --gpt2 "Hello world" --trace
    toklab decode -m bpe.json 72 101 108
    toklab compare "Tokenizers split text into pieces."
    toklab serve
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .base import Tokenizer, load
from .bpe import ByteLevelBPE
from .word import WordTokenizer
from .wordpiece import WordPiece

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def _read_corpus(paths: list[str]) -> str:
    return "\n".join(Path(p).read_text(encoding="utf-8") for p in paths)


def _train(kind: str, corpus: str, vocab_size: int, min_freq: int | None = None) -> Tokenizer:
    if kind == "word":
        return WordTokenizer.train(corpus, vocab_size=vocab_size, min_freq=min_freq or 1)
    if kind == "bpe":
        return ByteLevelBPE.train(corpus, vocab_size=vocab_size, min_frequency=min_freq or 2)
    if kind == "wordpiece":
        return WordPiece.train(corpus, vocab_size=vocab_size)
    raise ValueError(kind)


def _gpt2() -> ByteLevelBPE:
    return ByteLevelBPE.from_gpt2_files(DATA / "gpt2" / "encoder.json", DATA / "gpt2" / "vocab.bpe")


def _model(args) -> Tokenizer:
    if getattr(args, "gpt2", False):
        return _gpt2()
    if not args.model:
        sys.exit("error: pass -m MODEL.json or --gpt2")
    return load(args.model)


def cmd_train(args) -> None:
    corpus = _read_corpus(args.corpus)
    tok = _train(args.type, corpus, args.vocab_size, args.min_freq)
    tok.save(args.output)
    reason = getattr(tok, "stop_reason", "")
    print(f"trained {args.type} tokenizer: vocab_size={tok.vocab_size} {f'({reason})' if reason else ''}")
    print(f"saved to {args.output}")


def cmd_encode(args) -> None:
    tok = _model(args)
    text = args.text if args.text is not None else sys.stdin.read()
    if args.trace:
        print(json.dumps(tok.encode_trace(text), ensure_ascii=False, indent=1))
        return
    ids = tok.encode(text)
    print("tokens:", tok.tokenize(text))
    print("ids:   ", ids)
    print(f"{len(ids)} tokens for {len(text)} characters")


def cmd_decode(args) -> None:
    tok = _model(args)
    print(tok.decode(args.ids))


def cmd_compare(args) -> None:
    corpus = _read_corpus(args.corpus) if args.corpus else (DATA / "corpus.txt").read_text(encoding="utf-8")
    text = args.text
    models = [
        ("word", _train("word", corpus, args.vocab_size)),
        ("wordpiece", _train("wordpiece", corpus, args.vocab_size)),
        ("bpe", _train("bpe", corpus, args.vocab_size)),
        ("gpt2", _gpt2()),
    ]
    print(f"input: {text!r}  ({len(text)} chars, {len(text.encode())} bytes)\n")
    for name, tok in models:
        trace = tok.encode_trace(text)
        pieces = " | ".join(t.get("repr", t["text"]) for t in trace["tokens"])
        print(f"{name:<10} vocab={tok.vocab_size:<6} tokens={len(trace['ids']):<4} unk={trace['unk_count']}")
        print(f"           {pieces}\n")


def cmd_serve(args) -> None:
    from .site import build_site, serve

    out = build_site(Path(args.out))
    serve(out, args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="toklab", description="From-scratch LLM tokenizers.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train a tokenizer on text files")
    p.add_argument("corpus", nargs="+", help="UTF-8 text files")
    p.add_argument("--model", dest="type", choices=["word", "bpe", "wordpiece"], default="bpe")
    p.add_argument("--vocab-size", type=int, default=1000)
    p.add_argument("--min-freq", type=int, default=None)
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=cmd_train)

    for name, func, helptext in (("encode", cmd_encode, "text -> ids"), ("decode", cmd_decode, "ids -> text")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("-m", "--model", help="tokenizer JSON saved by `toklab train`")
        p.add_argument("--gpt2", action="store_true", help="use the GPT-2 vocabulary in data/gpt2")
        if name == "encode":
            p.add_argument("text", nargs="?", help="text to encode (default: stdin)")
            p.add_argument("--trace", action="store_true", help="print every pipeline stage as JSON")
        else:
            p.add_argument("ids", nargs="+", type=int)
        p.set_defaults(func=func)

    p = sub.add_parser("compare", help="compare word, WordPiece, BPE and GPT-2 on a text")
    p.add_argument("text")
    p.add_argument("--corpus", nargs="*", help="training files (default: data/corpus.txt)")
    p.add_argument("--vocab-size", type=int, default=600)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("serve", help="build and serve the web visualizer")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--out", default="_site")
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
