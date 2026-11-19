from pathlib import Path

import pytest

from toklab import WordPiece, load
from toklab.wordpiece import UNK

CORPUS = (Path(__file__).resolve().parent.parent / "data" / "corpus.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tok():
    return WordPiece.train(CORPUS, vocab_size=300, trace=True)


def test_vocab_size(tok):
    assert tok.vocab_size == 300


def test_initial_alphabet_uses_continuation_prefix():
    tok = WordPiece.train("cat cat", vocab_size=8)  # 5 specials + c, ##a, ##t
    assert {"c", "##a", "##t"} <= set(tok.vocab)


def test_score_prefers_pairs_that_rarely_appear_apart():
    # "q" is always followed by "u", while "a" and "b" also appear separately
    tok = WordPiece.train("qu qu ab ab ab a b a b ax bx", vocab_size=99, trace=True)
    first = tok.training_trace["steps"][0]
    assert tok.id_to_token(first["new_id"]) == "qu"


def test_greedy_longest_match(tok):
    pieces = tok.tokenize("playing")
    assert "".join(p.removeprefix("##") for p in pieces) == "playing"
    assert all(p.startswith("##") for p in pieces[1:])


def test_unknown_characters_become_unk(tok):
    assert tok.tokenize("cat 🤖") [-1] == UNK


def test_decode(tok):
    text = "the farmer walks to the field."
    assert tok.decode(tok.encode(text)) == text


def test_encode_trace(tok):
    trace = tok.encode_trace("Reading!")
    assert trace["normalized"] == "reading!"
    word = trace["pretokens"][0]
    assert word["steps"][0]["start"] == 0 and word["steps"][0]["tried"][0] == "reading"


def test_save_load(tok, tmp_path):
    tok.save(tmp_path / "wp.json")
    again = load(tmp_path / "wp.json")
    assert again.encode(CORPUS) == tok.encode(CORPUS)
