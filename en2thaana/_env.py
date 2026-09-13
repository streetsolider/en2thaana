"""Read a local .env, if there is one. No dependency, no magic.

Two inputs live outside this repository and are deliberately not shipped: the
evaluation reference and the place-name registry. Rather than hardcode where
they sit on one machine, both are read from environment variables, and this
loads them from a gitignored .env so local use needs no setup each time.

Absent both, the package still works: the gazetteer is empty and place names are
transcribed phonetically like any other word, and eval/ has nothing to score
against.
"""
from __future__ import annotations

import os
from pathlib import Path

_ENV = Path(__file__).resolve().parent.parent / ".env"


def load() -> None:
    if not _ENV.exists():
        return
    for line in _ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load()
