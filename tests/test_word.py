from toklab import WordTokenizer, load
from toklab.word import UNK

CORPUS = "the cat sat on the mat. the dog sat on the log. a cat and a dog!"


def test_train_orders_by_frequency():
    tok = WordTokenizer.train(CORPUS)
    assert tok.id_to_token(0) == "<pad>"
    assert tok.id_to_token(4) == "the"  # most frequent word comes right after specials
    assert tok.vocab_size == 4 + len(set(tok.pretokenize(CORPUS)))


def test_encode_decode_known_words():
    tok = WordTokenizer.train(CORPUS)
    ids = tok.encode("The cat sat on the mat.")
    assert tok.unk_id not in ids
    assert tok.decode(ids) == "the cat sat on the mat."


def test_unknown_words_map_to_unk():
    tok = WordTokenizer.train(CORPUS)
    ids = tok.encode("the zebra sat")
    assert ids[1] == tok.unk_id
    assert tok.decode(ids) == f"the {UNK} sat"


def test_vocab_size_and_min_freq():
    tok = WordTokenizer.train(CORPUS, vocab_size=8)
    assert tok.vocab_size == 8
    tok = WordTokenizer.train(CORPUS, min_freq=2)
    assert "mat" not in tok.vocab and "cat" in tok.vocab


def test_bos_eos():
    tok = WordTokenizer.train(CORPUS)
    ids = tok.encode("cat", add_bos=True, add_eos=True)
    assert ids[0] == tok.vocab["<bos>"] and ids[-1] == tok.vocab["<eos>"]
    assert tok.decode(ids) == "cat"


def test_trace():
    tok = WordTokenizer.train(CORPUS)
    trace = tok.encode_trace("The Zebra!")
    assert trace["normalized"] == "the zebra!"
    assert trace["pretokens"] == ["the", "zebra", "!"]
    assert trace["unk_count"] == 1


def test_save_load(tmp_path):
    tok = WordTokenizer.train(CORPUS)
    path = tmp_path / "word.json"
    tok.save(path)
    again = load(path)
    assert isinstance(again, WordTokenizer)
    assert again.encode(CORPUS) == tok.encode(CORPUS)
