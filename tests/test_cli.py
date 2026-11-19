from pathlib import Path

from toklab.cli import main

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus.txt"


def test_train_encode_decode(tmp_path, capsys):
    model = tmp_path / "bpe.json"
    main(["train", str(CORPUS), "--model", "bpe", "--vocab-size", "300", "-o", str(model)])
    main(["encode", "-m", str(model), "hello"])
    out = capsys.readouterr().out
    ids = out.split("ids:")[1].split("\n")[0].strip()
    main(["decode", "-m", str(model), *ids.strip("[]").split(", ")])
    assert capsys.readouterr().out.strip() == "hello"


def test_gpt2_encode(capsys):
    main(["encode", "--gpt2", "Hello world"])
    assert "[15496, 995]" in capsys.readouterr().out


def test_compare(capsys):
    main(["compare", "Tokenizers are unbelievable!", "--vocab-size", "300"])
    out = capsys.readouterr().out
    for name in ("word", "wordpiece", "bpe", "gpt2"):
        assert name in out
