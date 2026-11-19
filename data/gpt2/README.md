# GPT-2 vocabulary files

`encoder.json` (token -> id) and `vocab.bpe` (the 50,000 merge rules in rank order)
are OpenAI's published GPT-2 tokenizer files, downloaded from
`https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/`.
They are distributed under the license in `LICENSE` (Modified MIT, © 2019 OpenAI).

toklab uses them only to prove that its from-scratch BPE encoder reproduces GPT-2's
token ids exactly (`tests/test_gpt2_compat.py`) and to power the "GPT-2" mode in the
web visualizer. The code that reads and applies them is toklab's own.
