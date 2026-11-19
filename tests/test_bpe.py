from pathlib import Path

import pytest

from toklab import load
from toklab.bpe import ENDOFTEXT, ByteLevelBPE

CORPUS = (Path(__file__).resolve().parent.parent / "data" / "corpus.txt").read_text(encoding="utf-8")

TEXTS = [
    "Hello world!",
    "the tokenizer learns merges",
    "Unseen wörds, emoji 🤖 and 日本語 still work.",
    "  odd\twhitespace\n\n",
    "",
]


@pytest.fixture(scope="module")
def tok():
    return ByteLevelBPE.train(CORPUS, vocab_size=400, trace=True)


def test_vocab_size(tok):
    assert tok.vocab_size == 400
    assert len(tok.merges) >= 400 - 256 - 1


def test_first_merge_is_most_frequent_pair():
    # chunks: "ab" x1, " ab" x2, " cd" x1 -> pairs: (a,b)=3, (" ",a)=2, (" ",c)=1, (c,d)=1
    tok = ByteLevelBPE.train("ab ab ab cd", vocab_size=258, special_tokens=())
    # after merging (a,b), the pair (" ", "ab") occurs twice and is merged next
    assert tok.merges == [(b"a", b"b"), (b" ", b"ab")]
    assert tok.tokenize(" ab cd") == ["Ġab", "Ġ", "c", "d"]


def test_min_frequency_stops_training():
    tok = ByteLevelBPE.train("ab ab ab cd", vocab_size=300, special_tokens=())
    assert len(tok.merges) == 2 and "at least 2" in tok.stop_reason


def test_roundtrip(tok):
    for text in TEXTS:
        assert tok.decode(tok.encode(text)) == text


def test_compression(tok):
    ids = tok.encode(CORPUS)
    assert len(ids) < len(CORPUS.encode("utf-8")) * 0.6


def test_learns_common_words(tok):
    assert tok.tokenize(" the") == ["Ġthe"]


def test_special_tokens(tok):
    ids = tok.encode(f"a{ENDOFTEXT}b")
    assert tok.special_tokens[ENDOFTEXT] in ids
    assert tok.decode(ids) == f"a{ENDOFTEXT}b"
    assert tok.special_tokens[ENDOFTEXT] not in tok.encode(f"a{ENDOFTEXT}b", allowed_special=False)


def test_save_load(tok, tmp_path):
    path = tmp_path / "bpe.json"
    tok.save(path)
    again = load(path)
    assert isinstance(again, ByteLevelBPE)
    for text in TEXTS + [CORPUS]:
        assert again.encode(text) == tok.encode(text)


def test_gpt2_format_roundtrip(tok, tmp_path):
    tok.save_gpt2(tmp_path)
    again = ByteLevelBPE.from_gpt2_files(tmp_path / "encoder.json", tmp_path / "vocab.bpe")
    assert again.encode(CORPUS) == tok.encode(CORPUS)


def test_training_trace(tok):
    trace = tok.training_trace
    steps = trace["steps"]
    assert len(steps) == len(tok.merges)
    counts = [s["count"] for s in steps]
    assert counts == sorted(counts, reverse=True)  # BPE merge counts never increase
    for step in steps:
        assert step["top"][0][:2] == step["pair"]  # the chosen pair is the top candidate
    assert steps[0]["total_tokens"] < trace["corpus"]["bytes"]
    # replaying training gives exactly the same segmentation as encoding
    assert steps[-1]["total_tokens"] == len(tok.encode(CORPUS))


def test_encode_trace(tok):
    trace = tok.encode_trace("Hello world")
    assert [p["text"] for p in trace["pretokens"]] == ["Hello", " world"]
    assert trace["ids"] == tok.encode("Hello world")
    for p in trace["pretokens"]:
        if p["steps"]:
            assert p["steps"][-1]["parts"] == [t["repr"] for t in p["tokens"]]
