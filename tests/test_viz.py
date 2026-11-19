import json
from pathlib import Path

from toklab import viz

ROOT = Path(__file__).resolve().parent.parent
CORPUS = (ROOT / "data" / "corpus.txt").read_text(encoding="utf-8")


def test_viz_api_roundtrip():
    info = json.loads(viz.setup(CORPUS))
    assert set(info) == {"word", "bpe", "wordpiece"}
    gpt2 = ROOT / "data" / "gpt2"
    viz.load_gpt2((gpt2 / "encoder.json").read_text(encoding="utf-8"), (gpt2 / "vocab.bpe").read_text(encoding="utf-8"))
    for kind in ("word", "bpe", "wordpiece", "gpt2"):
        trace = json.loads(viz.encode(kind, "Hello tokenizer world!"))
        assert trace["ids"]
    rows = json.loads(viz.compare("Hello world"))
    assert [r["model"] for r in rows] == ["word", "wordpiece", "bpe", "gpt2"]
    trace = json.loads(viz.train("bpe", CORPUS, 300))
    assert len(trace["steps"]) >= 300 - 257 and trace["tokens"]


def test_build_site(tmp_path):
    from toklab.site import build_site

    out = build_site(tmp_path / "site")
    files = json.loads((out / "py" / "manifest.json").read_text())
    assert "viz.py" in files and "bpe.py" in files
    assert (out / "index.html").exists() and (out / "data" / "gpt2" / "vocab.bpe").exists()
