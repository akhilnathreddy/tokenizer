"""Build the static web visualizer into a folder and optionally serve it.

The page is plain HTML/CSS/JS plus this Python package, which the browser
runs with Pyodide.  ``build_site`` copies everything into one folder that can
be served by any static host (GitHub Pages in CI).

    python -m toklab.site _site      # build only
    toklab serve                     # build and serve on http://localhost:8000
"""

from __future__ import annotations

import http.server
import json
import shutil
import sys
from functools import partial
from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parents[1]


def build_site(out: Path) -> Path:
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "web", out)

    py = out / "py" / "toklab"
    py.mkdir(parents=True)
    files = sorted(p.name for p in PKG.glob("*.py") if p.name not in ("site.py", "__main__.py", "cli.py"))
    for name in files:
        shutil.copy2(PKG / name, py / name)
    (out / "py" / "manifest.json").write_text(json.dumps(files), encoding="utf-8")

    data = out / "data"
    (data / "gpt2").mkdir(parents=True)
    shutil.copy2(ROOT / "data" / "corpus.txt", data / "corpus.txt")
    for name in ("encoder.json", "vocab.bpe", "LICENSE"):
        shutil.copy2(ROOT / "data" / "gpt2" / name, data / "gpt2" / name)
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return out


def serve(directory: Path, port: int = 8000) -> None:
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"serving {directory} at http://localhost:{port}  (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    target = build_site(Path(sys.argv[1] if len(sys.argv) > 1 else "_site"))
    print(f"built site in {target}")
