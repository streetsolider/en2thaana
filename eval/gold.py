"""The evaluation reference: Hassan Hameed's English-Dhivehi dictionary.

LICENSING -- READ THIS
----------------------
Dr. Hassan Hameed's dictionary is 120,000 entries over 1,992 pages, self-funded
and commercially published in print. He has stated publicly that a free online
version will not be available until the costs are recouped.

This file therefore only ever READS his data, from a local copy kept outside
this repository. His entries are the held-out evaluation set for
this project and are never copied into data/, never written to dist/, and never
published. What this project ships is the rule engine and a dictionary generated
from open RP sources. Rules are not copyrightable; his entry list is.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

from en2thaana import _env  # noqa: F401  (loads .env on import)

# Local copy, never redistributed. Set EN2THAANA_GOLD to point at it.
HH = Path(os.environ.get(
    "EN2THAANA_GOLD",
    Path(__file__).resolve().parent.parent / "data" / "gold" / "english_dhivehi.tsv"))


def load_gold() -> list[tuple[str, str]]:
    """-> [(english, thaana)] for single-token pairs only. Multi-word headwords
    ('a la carte') and multi-word renderings are out of scope: this project maps
    one word to one pronunciation."""
    out = []
    with open(HH, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            en, th = r["english"].strip(), r["thaana"].strip()
            if en and th and " " not in en and " " not in th and en.isascii() and en.isalpha():
                out.append((en.lower(), th))
    return out
