"""Maps a saved tokenizer's ``kind`` to its class."""

from .bpe import ByteLevelBPE
from .word import WordTokenizer
from .wordpiece import WordPiece

REGISTRY = {
    WordTokenizer.kind: WordTokenizer,
    ByteLevelBPE.kind: ByteLevelBPE,
    WordPiece.kind: WordPiece,
}
