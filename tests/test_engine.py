"""Fixture tests, one per rule class. Every expected string is an attested
Hassan Hameed spelling, not an invention -- if a test fails, either the engine
regressed or the gold disagrees, and both are worth knowing.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from en2thaana import transcribe
from en2thaana.render import check
from en2thaana.thaana import parse, render

# (word, expected Thaana, what rule it exercises)
CASES = [
    ("brother",   "ބްރަދަރ",      "voiced th -> ދ, bare coda ރ"),
    ("decision",  "ޑިސިޜަން",      "/Z/ -> ޜ, the one dotted letter English needs"),
    ("anvil",     "އެންވިލް",       "schwa spelled <i> -> ި"),
    ("anthology", "އެންތޮލޮޖީ",    "schwa spelled <o> -> ޮ ; voiceless th -> ތ"),
    ("able",      "އޭބްލް",        "epenthetic schwa has no letter, so no fili"),
]


@pytest.mark.parametrize("word,expected,rule", CASES)
def test_matches_gold(word, expected, rule):
    assert transcribe(word).thaana == expected, rule


def test_no_epenthesis_in_initial_clusters():
    """English clusters are written out; Dhivehi's own ban does not apply."""
    th = transcribe("school").thaana
    assert th.startswith("\u0790\u07b0"),  "expected ސް.. not އިސް.."


def test_coda_r_is_bare_never_sukun():
    """ރ bare 1,475 times in the gold against ރް twice."""
    th = transcribe("brother").thaana
    assert th.endswith("\u0783")
    assert "\u0783\u07b0" not in th


def test_retroflex_t_not_dental():
    """English /t/ is ޓ; ތ is reserved for /T/."""
    assert "\u0793" in transcribe("beast").thaana
    assert "\u078c" not in transcribe("beast").thaana


@pytest.mark.parametrize("word", ["school", "computer", "strength", "decision",
                                  "brother", "able", "zuckerberg", "rhythm"])
def test_output_is_legal_thaana(word):
    r = transcribe(word)
    assert r.thaana and not check(r.thaana), r.problems


@pytest.mark.parametrize("word", ["school", "anthology", "brownie", "able"])
def test_parser_roundtrips(word):
    th = transcribe(word).thaana
    assert render(parse(th)) == th


def test_forbidden_letters_never_emitted():
    """ޅ and ޏ are Dhivehi-only: 0 occurrences in 8,662 gold words."""
    from en2thaana.ipa import load_en_uk
    for w in list(load_en_uk())[:3000]:
        th = transcribe(w).thaana
        if th:
            assert "\u0785" not in th and "\u078f" not in th, w


# --- the LOT vowel, and the confidence guard on the orthographic feature ------

@pytest.mark.parametrize("word,expected", [
    ("was",   "ވޮޒް"),      # /wɒz/ -- LOT, and a FINAL /z/, not /s/
    ("wash",  "ވޮޝް"),
    ("what",  "ވޮޓް"),
    ("want",  "ވޮންޓް"),
    ("watch", "ވޮޗް"),
])
def test_lot_vowel_spelled_a(word, expected):
    """/ɒ/ written <a> after /w/ takes ޮ, attested in the gold as awash
    ->އެވޮޝް, backwash ->ބެކްވޮޝް, brainwash ->ބްރެއިންވޮޝް.

    The orthographic cell for (/ɒ/, <a>) is only 33% confident (7 of 21) yet was
    overriding the phoneme table's 84% ޮ, giving ވާޒް for `was`. An orthographic
    cell may only displace a rule it is more confident than."""
    assert transcribe(word).thaana == expected


def test_was_ends_in_z_not_s():
    """ވޯސް and ވާސް both read as `wass`. The word ends in /z/."""
    th = transcribe("was").thaana
    assert th.endswith("ޒް"), th
    assert "ސ" not in th


# --- source selection: which lexicon is trusted for what ----------------------

@pytest.mark.parametrize("word,expected", [
    ("stochastic", "ސްޓޮކެސްޓިކް"),   # Greek <ch> is /k/, never /tʃ/
    ("chaos",      "ކޭއޮސް"),
    ("chemistry",  "ކެމިސްޓްރީ"),
])
def test_greek_ch_is_k(word, expected):
    """ipa-dict's en_UK is espeak-generated and guessed /tʃ/ for `stochastic`.
    Britfone and CMUdict both have /k/, and Britfone is consulted first."""
    assert transcribe(word).thaana == expected


@pytest.mark.parametrize("word,letter", [
    ("absorb", "ޒ"),        # /z/, not /s/
    ("acquisition", "ޒ"),   # /z/, not /s/
    ("amuse", "ޒ"),         # /z/, not /s/
    ("algorithm", "ދ"),     # /ð/, not /θ/ -- gold: އެލްގަރިދަމް
])
def test_curated_consonants_beat_generated_ones(word, letter):
    """292 of the 12,998 words in both RP sources disagree on consonants, and
    the espeak-derived one is wrong in most: /s/ for /z/, /θ/ for /ð/, and a
    missing /ŋ/ before /k/."""
    assert letter in transcribe(word).thaana


@pytest.mark.parametrize("word,expected", [
    ("bus", "ބަސް"), ("blood", "ބްލަޑް"), ("bug", "ބަގް"),
])
def test_strut_is_not_flattened_to_schwa(word, expected):
    """Britfone writes STRUT as ɐ, where espeak uses ɐ for a reduced vowel.
    Reading Britfone's ɐ as schwa flattened 333 STRUT words."""
    assert transcribe(word).thaana == expected


@pytest.mark.parametrize("word,expected", [
    ("was", "ވޮޒް"), ("to", "ޓޫ"), ("of", "އޮވް"),
    ("you", "ޔޫ"), ("and", "އެންޑް"),
])
def test_function_words_use_the_strong_form(word, expected):
    """A pronunciation guide gives the citation form. Britfone stores the weak
    form for 18 function words -- to /tə/, was /wəz/ -- which is how they sound
    in connected speech but not what a dictionary prints."""
    assert transcribe(word).thaana == expected


# --- the /juː/ glide ---------------------------------------------------------

@pytest.mark.parametrize("word,expected", [
    # These four also match Hassan Hameed exactly -- they are among the 43% of
    # his /juː/ spellings that use ޔ.
    ("music",     "މިޔުޒިކް"),
    ("amuse",     "އެމިޔުޒް"),
    ("acute",     "އެކިޔުޓް"),
    ("calculate", "ކެލްކިޔުލޭޓް"),
    # These deliberately DIFFER from him: he writes ކޮމްޕިއުޓަރ with alifu.
    # The carrier is a speaker's decision on a question his own data leaves
    # open (124 alifu vs 94 yaviyani), so the disagreement is intended.
    ("computer",  "ކޮމްޕިޔުޓަރ"),
    ("ambulance", "އެމްބިޔުލަންސް"),
    ("altitude",  "އެލްޓިޓިޔުޑް"),
])
def test_glide_uses_the_chosen_carrier(word, expected):
    """/j/ between a consonant and a vowel is an i-fili on the preceding letter,
    not a consonant of its own, and the vowel that follows opens a ޔ carrier."""
    assert transcribe(word).thaana == expected


@pytest.mark.parametrize("word", ["computer", "music", "beauty", "amuse",
                                  "calculate", "regular", "popular"])
def test_glide_never_writes_sukun_before_yaviyani(word):
    """The old output was ކޮމްޕްޔޫޓަރ -- sukun then ޔ. That spelling appears in
    9 gold words against 218 for the i-fili form."""
    th = transcribe(word).thaana
    assert "\u07B0\u0794" not in th, th


@pytest.mark.parametrize("word", ["computer", "music", "amuse", "acute"])
def test_glide_vowel_is_short(word):
    """The /uː/ after the glide is written short ު, not long ޫ -- 193 gold
    words against 22."""
    th = transcribe(word).thaana
    assert "\u07A8\u0787\u07AB" not in th and "\u07A8\u0794\u07AB" not in th, th
