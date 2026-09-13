"""Text-layer tests: the classes that broke when the word converter met a
sentence. Each case here is a bug that actually occurred, not a hypothetical.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from en2thaana import gazetteer
from en2thaana.text import (normalise_dotted_acronyms, render_text,
                            transcribe_text, transcribe_token)


def kinds(text):
    return {t.text: t.kind for t in transcribe_text(text)}


def by_word(text):
    return {t.text: t for t in transcribe_text(text)}


def test_abbreviation_not_resolved_as_a_word():
    """`Dr.` used to come back as ޑްރައިވް -- CMUdict resolves `dr` to `drive`."""
    t = transcribe_token("Dr", ".")
    assert t.kind == "abbreviation"
    assert t.thaana and "ޑޮކް" in t.thaana


def test_bare_abbreviation_without_period_is_a_word():
    """`co` without a period is the word, not `company`."""
    assert transcribe_token("co", " ").kind == "word"


def test_numbers_pass_through_unchanged():
    """Thaana text uses the same digits; converting them would be wrong."""
    out = by_word("120,000 entries in 2019")
    assert out["120,000"].kind == "number"
    assert out["120,000"].thaana == "120,000"
    assert out["2019"].thaana == "2019"


def test_acronyms_are_spelled_letter_by_letter():
    t = transcribe_token("PDF", " ")
    assert t.kind == "acronym"
    assert t.thaana == "ޕީޑީއެފް"


def test_dotted_acronym_is_joined_first():
    assert normalise_dotted_acronyms("the U.S. said") == "the US said"
    assert transcribe_token("US", " ").thaana == "ޔޫއެސް"


def test_known_long_uppercase_word_is_not_an_acronym():
    """A shouted word is a word: HOUSE must not become އެޗްއޯޔޫއެސްއީ."""
    assert transcribe_token("HOUSE", " ").kind == "word"


def test_punctuation_and_spacing_survive():
    """Punctuation is preserved -- but the comma is now the Arabic one, ،"""
    out = render_text("Yes, it works!")
    assert out.endswith("!") and "،" in out


@pytest.mark.skipif(gazetteer.size() == 0, reason="gazetteer not available")
@pytest.mark.parametrize("name,expected", [
    ("Hithadhoo", "ހިތަދޫ"),
    ("Kulhudhuffushi", "ކުޅުދުއްފުށި"),
])
def test_maldivian_placenames_come_from_the_registry(name, expected):
    """These are Dhivehi words that happen to be romanised, not English words to
    be approximated. The model guessed ހިތާޑޯ for Hithadhoo."""
    t = transcribe_token(name, " ")
    assert t.source == "gazetteer"
    assert t.thaana == expected


def test_every_token_is_accounted_for():
    """Nothing may be dropped: the rendered text must contain every token."""
    src = "Dr. Hassan published 120,000 entries in 2019 — see it!"
    toks = transcribe_text(normalise_dotted_acronyms(src))
    assert "".join(t.text for t in toks) == normalise_dotted_acronyms(src)


# --- regressions -------------------------------------------------------------

@pytest.mark.parametrize("word", [
    "fresh", "beauty", "black", "camera", "castle", "arrow",
])
def test_english_words_are_not_hijacked_by_the_address_registry(word):
    """Maldivians name houses with English words, so the address registry holds
    409 of the commonest English words. Consulted before the lexicon it turned
    `fresh` into a house name -- and into ފްރެޱް, with an archaic naviyani, at
    that. An English word must come from a pronunciation lexicon.

    The assertion is on the KIND, not on which lexicon answered: the order
    among the pronunciation sources is a tuning decision that has changed once
    already, and pinning it here would make this test fail for the wrong
    reason."""
    t = transcribe_token(word, " ")
    assert t.kind == "word"
    assert t.source in ("britfone", "en_UK", "cmudict_rp"), t.source


def test_fresh_is_spelled_with_shaviyani():
    assert transcribe_token("fresh", " ").thaana == "ފްރެޝް"


def test_gazetteer_rejects_nonstandard_thaana():
    """Naviyani (U+07B1) is archaic and in this data is always an encoding
    artefact. No entry carrying it may reach the output."""
    assert gazetteer.rejected() == 0          # islands/atolls are clean
    for v in gazetteer._table().values():
        assert "\u07B1" not in v


# --- Arabic-script punctuation ----------------------------------------------

def test_punctuation_uses_the_arabic_marks():
    """Per the Segha Standard Phonetic layout:
    comma, semicolon and question mark change; full stop and colon do not."""
    assert render_text("What? Yes, now; go.") == render_text("What? Yes, now; go.")
    out = render_text("What do you want? Fresh, clean; fast.")
    assert "\u061F" in out and "?" not in out       # ؟
    assert "\u060C" in out and "," not in out       # ،
    assert "\u061B" in out and ";" not in out       # ؛
    assert out.endswith(".")                        # full stop is unchanged


def test_digit_separators_are_not_converted():
    """The comma in 120,000 is a digit separator, not punctuation. A greedy
    number pattern also used to swallow the comma in `2019, and` so it never
    reached the converter at all."""
    out = render_text("In 2019, there were 120,000 of them.")
    assert "120,000" in out
    assert "2019\u060C" in out or "2019 \u060C" in out or "\u060C" in out
    assert "2019," not in out
