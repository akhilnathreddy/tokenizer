"""Reversible mapping between raw bytes and printable Unicode characters.

Byte-level BPE works on UTF-8 bytes, but vocab files are text.  GPT-2 maps each
of the 256 byte values to a visible character: printable ASCII and Latin-1
bytes map to themselves, the rest (control chars, space, ...) are shifted to
code points 256+.  That is why a leading space shows up as "Ġ" (byte 0x20 ->
U+0120) and a newline as "Ċ".
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codes = printable[:]
    shift = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            codes.append(256 + shift)
            shift += 1
    return {b: chr(c) for b, c in zip(printable, codes)}


@lru_cache(maxsize=1)
def unicode_to_bytes() -> dict[str, int]:
    return {c: b for b, c in bytes_to_unicode().items()}


def bytes_to_repr(data: bytes) -> str:
    """bytes -> GPT-2 style printable string, e.g. b" hi" -> "Ġhi"."""
    table = bytes_to_unicode()
    return "".join(table[b] for b in data)


def repr_to_bytes(text: str) -> bytes:
    """Inverse of :func:`bytes_to_repr`."""
    table = unicode_to_bytes()
    return bytes(table[c] for c in text)
