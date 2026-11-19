"""Maps a saved tokenizer's ``kind`` to its class."""

from .bpe import ByteLevelBPE
from .word import WordTokenizer

REGISTRY = {
    WordTokenizer.kind: WordTokenizer,
    ByteLevelBPE.kind: ByteLevelBPE,
}
