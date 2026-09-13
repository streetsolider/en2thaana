"""The shipped dictionary must agree with the live engine.

dist/en_thaana.jsonl and the web assets are generated, and a generated artifact
goes stale the moment a rule changes. It happened: the first web build shipped
ކޮމްޕްޔޫޓަރ and ވާޒް, spellings three fixes out of date, and nothing caught it
because every unit test exercised the engine rather than the build output.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from en2thaana import transcribe

DIST = ROOT / "dist" / "en_thaana.jsonl"
WEB = ROOT / "web" / "public" / "dict.txt"

SAMPLE = ["computer", "was", "stochastic", "music", "beauty", "fresh",
          "brother", "decision", "able", "anthology", "absorb", "algorithm"]


@pytest.mark.skipif(not DIST.exists(), reason="dictionary not built")
def test_dist_matches_the_engine():
    want = {w: transcribe(w).thaana for w in SAMPLE}
    got = {}
    for line in DIST.open(encoding="utf-8"):
        r = json.loads(line)
        if r["word"] in want:
            got[r["word"]] = r["thaana"]
    stale = {w: (got.get(w), want[w]) for w in want if got.get(w) != want[w]}
    assert not stale, ("dist/en_thaana.jsonl is stale -- run "
                       "`python -m en2thaana --build dist/en_thaana.jsonl`: " + str(stale))


@pytest.mark.skipif(not WEB.exists(), reason="web assets not built")
def test_web_assets_match_dist():
    web = {}
    for line in WEB.read_text(encoding="utf-8").split("\n"):
        i = line.find("\t")
        if i > 0 and line[:i] in SAMPLE:
            web[line[:i]] = line[i + 1:]
    want = {w: transcribe(w).thaana for w in SAMPLE}
    stale = {w: (web.get(w), want[w]) for w in want if web.get(w) != want[w]}
    assert not stale, "web/public/dict.txt is stale -- run `python build_web.py`: " + str(stale)
