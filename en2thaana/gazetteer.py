"""Maldivian proper nouns: use the attested Thaana spelling, never a phonetic guess.

WHY THIS EXISTS
---------------
`Hithadhoo` is not an English word being approximated -- it is a Dhivehi word
that happens to have been romanised, and it has one correct spelling: ހިތަދޫ.
Asked to guess, the OOV model produced ހިތާޑޯ, a faithful rendering of how an
English speaker would read those letters, and wrong.

WHY IT IS CONSULTED *AFTER* THE ENGLISH LEXICON, NOT BEFORE
-----------------------------------------------------------
Maldivians name houses with English words, so the address registry contains 409
of the 20,000 commonest English words -- fresh, beauty, best, black, brown,
camera, castle, cinema, arrow. Consulted first, the gazetteer hijacked every one
of them: `fresh` in an English sentence came back as the house name, not the
word. So the order is lexicon -> gazetteer -> model: the gazetteer answers only
for names no English dictionary knows, which is exactly the set it is good for.

WHY ADDRESSES ARE EXCLUDED BY DEFAULT
-------------------------------------
Beyond the collisions, 422 of the 7,485 address rows carry ޱ (naviyani, U+07B1)
where ޝ (shaviyani) belongs -- `nishaan` as ނިޱާން, `shaadhee` as ޱާދީ. Every
affected row is a romanised `sh`, which is the signature of a legacy pre-Unicode
encoding conversion, not a dialect spelling. Naviyani was dropped from standard
Dhivehi orthography in the 1950s. islands.tsv and atolls.tsv are clean.

Rather than trust-and-repair, every entry is validated on load and anything
outside the standard Thaana block is rejected, so corrupt source data cannot
reach the output even if a new file is added later.

The registry data is read in place from a local copy and is not redistributed
with this package.
"""
from __future__ import annotations

import csv
from functools import lru_cache
import os
from pathlib import Path

from en2thaana import _env  # noqa: F401  (loads .env on import)

# Local, unredistributed registry data. Override with EN2THAANA_GAZETTEER;
# absent, the gazetteer is simply empty and the engine transcribes place names
# phonetically like any other word.
GAZ = Path(os.environ.get("EN2THAANA_GAZETTEER",
                          Path(__file__).resolve().parent.parent / "data" / "gazetteer"))

# Standard modern Thaana: the 24 base letters, the 14 dotted letters, the 10
# fili and sukun. NOT U+07B1 naviyani, which is archaic and here is always an
# encoding artefact.
THAANA_OK = {chr(c) for c in range(0x0780, 0x07B1)}
NAVIYANI = "ޱ"

# islands and atolls only. addresses.tsv is excluded: see the module docstring.
SOURCES = ("islands.tsv", "atolls.tsv")


# Atoll codes are written as abbreviations and legitimately end in a period
# (ހދ.), and a few island names are two words.
PUNCT_OK = {" ", "."}


def _valid(thaana: str) -> bool:
    return bool(thaana) and all(c in THAANA_OK or c in PUNCT_OK for c in thaana)


def _load_tsv(name: str) -> tuple[dict[str, str], int]:
    path = GAZ / name
    out: dict[str, str] = {}
    rejected = 0
    if not path.exists():
        return out, 0
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            lat = (r.get("latin") or "").strip().lower()
            th = (r.get("thaana") or "").strip()
            if not lat or not th:
                continue
            if not (lat.isascii() and lat.replace(" ", "").isalpha()):
                continue
            if not _valid(th):
                rejected += 1
                continue
            out.setdefault(lat, th)
    return out, rejected


@lru_cache(maxsize=1)
def _build() -> tuple[dict[str, str], int]:
    table: dict[str, str] = {}
    rejected = 0
    for name in SOURCES:
        rows, bad = _load_tsv(name)
        table.update(rows)
        rejected += bad
    return table, rejected


def _table() -> dict[str, str]:
    return _build()[0]


def rejected() -> int:
    """Entries dropped for containing characters outside standard Thaana."""
    return _build()[1]


def lookup(word: str) -> str | None:
    """Attested Thaana spelling for a Maldivian proper noun, or None.

    Callers must try the English pronunciation lexicon FIRST -- see the module
    docstring. This function does not know whether its key is also an ordinary
    English word, and several hundred of them are."""
    return _table().get(word.strip().lower())


def size() -> int:
    return len(_table())
