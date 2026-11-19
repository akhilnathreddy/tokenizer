"""Maps a saved tokenizer's ``kind`` to its class."""

from .word import WordTokenizer

REGISTRY = {
    WordTokenizer.kind: WordTokenizer,
}
