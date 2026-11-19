"""Regenerate tests/fixtures/gpt2_expected.json using OpenAI's tiktoken.

tiktoken is only used here, offline, to produce reference ids; the toklab
tests then compare against the saved file without needing tiktoken.

    pip install tiktoken
    python scripts/make_gpt2_fixture.py
"""

import json
from pathlib import Path

import tiktoken

SENTENCES = [
    "Hello world!",
    "The quick brown fox jumps over the lazy dog.",
    "I'm sure they'll say it's fine, but we've seen what you'd done.",
    "Tokenization isn't magic: it's just byte pair encoding.",
    "   leading spaces, trailing spaces   ",
    "Line one\nLine two\n\n\tTabbed line\r\n",
    "Numbers: 3.14159, 1,000,000 and 2026-11-19.",
    "Emoji 🤖🚀🎉 and accents: café, naïve, résumé.",
    "日本語のテキスト、中文文本、한국어 텍스트",
    "हिन्दी और తెలుగు भाषाएँ",
    "Ελληνικά και русский текст",
    "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
    "<html><body class=\"main\">Hi &amp; bye</body></html>",
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
    "HTTP://EXAMPLE.COM/path?query=value&x=1#frag",
    "Before <|endoftext|> after",
    "''''''''''''''''",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "   odd   whitespace　here",
    "",
]


def main() -> None:
    enc = tiktoken.get_encoding("gpt2")
    cases = [{"text": s, "ids": enc.encode(s, allowed_special="all")} for s in SENTENCES]
    out = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "gpt2_expected.json"
    out.write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
