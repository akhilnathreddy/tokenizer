import random

import pytest

from toklab.byte_unicode import bytes_to_repr, bytes_to_unicode, repr_to_bytes
from toklab.normalize import Normalizer
from toklab.pretokenize import GPT2_PATTERN, gpt2_split, split_words

SAMPLES = [
    "Hello world!",
    "I'm here, you're there; they'll go and we've won. He'd say it's 's",
    "  leading and trailing spaces   ",
    "tabs\tand\nnewlines\n\n  indented",
    "numbers 12345 and 3.14 and ½ ²",
    "emoji 🤖🚀 and symbols @#$%^&*()",
    "日本語のテキスト and Ελληνικά and हिन्दी",
    "code: def f(x):\n    return x**2  # comment",
    "'quoted' \"double\" ''' '''s",
    " non-breaking　ideographic space",
    "",
    " ",
    "a",
]


def test_gpt2_split_is_lossless():
    for s in SAMPLES:
        assert "".join(gpt2_split(s)) == s


def test_gpt2_split_known_example():
    assert gpt2_split("Hello world, I'm 42!") == ["Hello", " world", ",", " I", "'m", " 42", "!"]
    assert gpt2_split("a  b") == ["a", " ", " b"]
    assert gpt2_split("x\n\ny") == ["x", "\n", "\n", "y"]
    assert gpt2_split("x\n\n") == ["x", "\n\n"]


regex = pytest.importorskip("regex")


@pytest.mark.parametrize("text", SAMPLES)
def test_gpt2_split_matches_regex(text):
    assert gpt2_split(text) == regex.findall(GPT2_PATTERN, text)


def test_gpt2_split_matches_regex_fuzz():
    alphabet = "ab Z9 '\t\n.,!é字🤖 ½_-'stmdlrve"
    rng = random.Random(0)
    for _ in range(3000):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 25)))
        assert gpt2_split(s) == regex.findall(GPT2_PATTERN, s), repr(s)


def test_split_words():
    assert split_words("Hello, world! It's 2026.") == [
        "Hello", ",", "world", "!", "It", "'", "s", "2026", ".",
    ]


def test_byte_unicode_roundtrip():
    table = bytes_to_unicode()
    assert len(table) == 256 and len(set(table.values())) == 256
    assert table[ord(" ")] == "Ġ" and table[ord("\n")] == "Ċ"
    data = "héllo 🤖\n".encode()
    assert repr_to_bytes(bytes_to_repr(data)) == data


def test_normalizer():
    n = Normalizer(form="NFKC", lowercase=True, strip_accents=True)
    assert n("Café ＡＢＣ") == "cafe abc"
    assert Normalizer.from_dict(n.to_dict()) == n
