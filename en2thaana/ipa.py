"""Stage 1: English word -> RP IPA.

WHY RP
------
Hassan Hameed's English-Dhivehi dictionary -- the only attested body of English
words written in Thaana, and this project's evaluation reference -- states that
its pronunciations use Received Pronunciation. CMUdict is General American, so
it cannot be the primary source: it writes `car` as /kar/ where RP needs /kA:/.

SOURCE ORDER (most trusted first)
---------------------------------
1. ipa-dict en_UK      65k entries, RP, already IPA.        source="en_UK"
2. Britfone            16k entries, RP, with stress marks.  source="britfone"
3. CMUdict + a deterministic GenAm->RP transform.           source="cmudict_rp"

Every lookup returns the source that produced it, so the share carried by the
transform (and later by the OOV model) is always measurable rather than assumed.
Nothing here learns anything; steps 1-3 are lookup and rule.
"""
from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------- ARPAbet -> IPA
# General American values. AH is split by stress: stressed AH1 is STRUT /V/,
# unstressed AH0 is schwa. ER likewise (NURSE vs lettER).
ARPA_IPA = {
    "AA": "\u0251", "AE": "\u00e6", "AO": "\u0254", "AW": "a\u028a", "AY": "a\u026a",
    "B": "b", "CH": "t\u0283", "D": "d", "DH": "\u00f0", "EH": "\u025b",
    "EY": "e\u026a", "F": "f", "G": "\u0261", "HH": "h", "IH": "\u026a",
    "IY": "i", "JH": "d\u0292", "K": "k", "L": "l", "M": "m", "N": "n",
    "NG": "\u014b", "OW": "o\u028a", "OY": "\u0254\u026a", "P": "p", "R": "r",
    "S": "s", "SH": "\u0283", "T": "t", "TH": "\u03b8", "UH": "\u028a",
    "UW": "u", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "\u0292",
}
_STRESS = re.compile(r"[012]$")

VOWELS = set("\u0251\u00e6\u0254\u025b\u026a i u \u028a \u028c \u0259 \u025c \u025a e o a".split()) | set(
    "\u0251\u00e6\u0254\u025b\u026aiu\u028a\u028c\u0259\u025c\u025aeoa"
)


def arpabet_to_ipa(phones: list[str]) -> str:
    """ARPAbet with stress digits -> GenAm IPA. Stress is dropped: Thaana has no
    stress mark, so carrying it further would only be discarded downstream."""
    out = []
    for p in phones:
        stress = p[-1] if p and p[-1] in "012" else ""
        base = _STRESS.sub("", p)
        if base == "AH":
            out.append("\u028c" if stress == "1" else "\u0259")
        elif base == "ER":
            out.append("\u025c" if stress == "1" else "\u025a")
        else:
            out.append(ARPA_IPA.get(base, ""))
    return "".join(out)


# ---------------------------------------------------------- GenAm -> RP transform
# Ordered: the r-coloured sequences must go before the general non-prevocalic
# /r/ deletion, or `car` loses its r before /A/ has been lengthened.
_RHOTIC_PAIRS = [
    ("\u0251r", "\u0251\u02d0"),        # ar   -> A:    car
    ("\u0254r", "\u0254\u02d0"),        # Or   -> O:    for
    ("\u025cr", "\u025c\u02d0"), ("\u025c", "\u025c\u02d0"),   # 3`  -> 3:   bird
    ("\u025ar", "\u0259"), ("\u025a", "\u0259"),               # @`  -> @    letter
    ("\u026ar", "\u026a\u0259"),        # Ir   -> I@    here
    ("\u025br", "e\u0259"),             # Er   -> e@    hair
    ("\u028ar", "\u028a\u0259"),        # Ur   -> U@    tour
]


def genam_to_rp(ipa: str) -> str:
    """Deterministic GenAm -> RP. Covers the systematic (non-lexical) differences
    only. BATH-broadening (`bath` /baeT/ -> /bA:T/) is lexically conditioned, not
    predictable from the segment string, so it is deliberately NOT applied --
    guessing it would corrupt more words than it fixes."""
    s = ipa
    for a, b in _RHOTIC_PAIRS:
        s = s.replace(a, b)
    # non-prevocalic /r/ is not pronounced in RP
    s = re.sub(r"r(?![\u0251\u00e6\u0254\u025b\u026aiu\u028a\u028c\u0259\u025ceoa])", "", s)
    s = s.replace("o\u028a", "\u0259\u028a")       # GOAT  oU -> @U
    s = s.replace("\u0251\u02d0", "\u0251\u02d0")  # keep
    # LOT is unrounded in GenAm, rounded in RP -- but only when short
    s = re.sub(r"\u0251(?!\u02d0)", "\u0252", s)
    s = re.sub(r"u(?!\u02d0)", "u\u02d0", s)
    s = re.sub(r"i(?!\u02d0|\u0259)", "i\u02d0", s)
    return s


# --------------------------------------------------------------------- loaders
def _strip_ipa(s: str) -> str:
    """Drop slashes, stress marks, syllable dots, length-neutral diacritics."""
    s = s.strip().strip("/[]")
    for ch in "\u02c8\u02cc.\u200d\u0361":
        s = s.replace(ch, "")
    return s.strip()


@lru_cache(maxsize=1)
def load_en_uk() -> dict[str, str]:
    out = {}
    p = DATA / "ipa_en_UK.txt"
    for line in p.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        w, ipas = line.split("\t", 1)
        first = ipas.split(",")[0]
        v = _strip_ipa(first)
        if v:
            out.setdefault(w.strip().lower(), v)
    return out


@lru_cache(maxsize=1)
def load_britfone() -> dict[str, str]:
    """Britfone, with its own reading of U+0250 restored.

    The two sources use the same symbol for different vowels. In espeak's en_UK,
    ɐ is a reduced, schwa-like vowel. In Britfone it is STRUT: blood /blɐd/,
    bus /bɐs/, bug /bɐɡ/, none of which are reduced. Applying en_UK's reading to
    Britfone flattened 333 STRUT words into schwa. Normalise it here, before the
    shared normaliser gets to it."""
    out = {}
    p = DATA / "britfone.csv"
    for row in csv.reader(p.read_text(encoding="utf-8").splitlines()):
        if len(row) < 2:
            continue
        w = re.sub(r"\(\d+\)$", "", row[0].strip()).lower()
        v = _strip_ipa(" ".join(row[1:]).replace(" ", ""))
        v = v.replace("ɐ", "ʌ")      # ɐ -> ʌ  (STRUT, not schwa)
        if w and v:
            out.setdefault(w, v)
    return out


@lru_cache(maxsize=1)
def load_cmudict() -> dict[str, list[str]]:
    out = {}
    p = DATA / "cmudict.dict"
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        w = re.sub(r"\(\d+\)$", "", parts[0]).lower()
        out.setdefault(w, parts[1:])
    return out


# ---------------------------------------------------------------------- lookup
def _lone_schwa(ipa: str) -> bool:
    """True when the string has exactly one vowel and that vowel is schwa."""
    vs = [p for p in phonemes(ipa) if is_vowel_phoneme(p)]
    return len(vs) == 1 and vs[0] == "ə"


def lookup(word: str) -> tuple[str | None, str]:
    """-> (RP IPA, source). source is one of britfone / en_UK / cmudict_rp / none.

    BRITFONE FIRST, DESPITE BEING THE SMALLER LIST
    ----------------------------------------------
    ipa-dict's en_UK is espeak-generated; Britfone is human-curated. Where both
    have a word (12,998 of them) their consonant skeletons disagree 292 times,
    and en_UK is wrong in the great majority: it writes /s/ for /z/ (absorb,
    acquisition, amuse), /T/ for /D/ (algorithm), and misses /N/ before /k/
    (anchor). The gold data settles it independently -- Hassan Hameed writes
    algorithm as އެލްގަރިދަމް with ދ = /D/, which is Britfone's reading.

    Vowel and rhoticity differences between the two are legitimate accent
    variation. Consonant identity is not, so the curated list wins."""
    w = word.strip().lower()
    bf = load_britfone()
    uk = load_en_uk()
    if w in bf:
        # Britfone stores the WEAK form of 18 function words -- to /t@/, was
        # /w@z/, of /@v/, your /j@/. That is how they sound in connected speech,
        # but a pronunciation guide gives the citation form: a dictionary prints
        # `to /tu:/`. Where a monosyllable reduces to a lone schwa here and
        # another source keeps a full vowel, take the full vowel.
        if _lone_schwa(bf[w]) and w in uk and not _lone_schwa(uk[w]):
            return uk[w], "en_UK"
        return bf[w], "britfone"
    if w in uk:
        return uk[w], "en_UK"
    cmu = load_cmudict()
    if w in cmu:
        return genam_to_rp(arpabet_to_ipa(cmu[w])), "cmudict_rp"
    return None, "none"


# ------------------------------------------------------- normalisation + tokens
# Measured over both sources, not assumed. ipa-dict/Britfone are espeak-derived
# and use three non-standard spellings; everything else in their 44- and
# 37-symbol inventories is already canonical.
_NORMALISE = {
    "\u0279": "r",        # turned r -> r      (20,796 in en_UK)
    "\u0250": "\u0259",   # turned a -> schwa  ( 7,596)
    "g": "\u0261",        # plain g -> script g (Britfone)
    "\u1d7b": "\u026a",   # small cap I with stroke -> I
}
# Residue appearing 1-6 times each: foreign loans espeak did not anglicise.
_DROP = "\u0303\u02b2\u026c\u0320\u0348 x"

# Longest-match first. The affricates and diphthongs MUST precede their parts:
# both sources write /tS/ as t + S, and mapping those separately would give
# T + SH (Thaana ޓ + ޝ) instead of CH (ޗ).
PHONEMES = [
    # affricates
    "t\u0283", "d\u0292",
    # long vowels
    "i\u02d0", "\u0251\u02d0", "\u0254\u02d0", "u\u02d0", "\u025c\u02d0",
    # diphthongs
    "e\u026a", "a\u026a", "\u0254\u026a", "\u0259\u028a", "a\u028a",
    "\u026a\u0259", "e\u0259", "\u028a\u0259",
    # consonants
    "p", "b", "t", "d", "k", "\u0261", "f", "v", "\u03b8", "\u00f0",
    "s", "z", "\u0283", "\u0292", "h", "m", "n", "\u014b", "l", "r", "w", "j",
    # short / weak vowels
    "\u026a", "e", "\u025b", "\u00e6", "\u028c", "\u0252", "\u028a",
    "\u0259", "i", "u", "a", "\u0251", "\u0254", "\u025c",
]
_PHON_SORTED = sorted(PHONEMES, key=len, reverse=True)


def normalise(ipa: str) -> str:
    out = []
    for ch in ipa:
        if ch in _DROP:
            continue
        out.append(_NORMALISE.get(ch, ch))
    return "".join(out)


def phonemes(ipa: str) -> list[str]:
    """Split an RP IPA string into phonemes, longest match first.

    Unknown characters are passed through as single-character phonemes so they
    surface in the induction report rather than being silently dropped."""
    s = normalise(ipa)
    out, i = [], 0
    while i < len(s):
        for p in _PHON_SORTED:
            if s.startswith(p, i):
                out.append(p)
                i += len(p)
                break
        else:
            out.append(s[i])
            i += 1
    return out


# ------------------------------------------------------------------- backends
# Which lexicon supplies the phoneme string is an empirical question, not a
# stylistic one, so it is switchable and measured. See eval/backends.py.
#
#   rp      ipa-dict en_UK -> Britfone -> CMUdict+transform.  Non-rhotic.
#   genam   CMUdict as-is.                                    Rhotic.
#   rhotic  en_UK RP vowels, with /r/ restored wherever the English spelling
#           has an <r> that RP dropped. Hassan Hameed's renderings are visibly
#           spelling-influenced -- he writes ރ in `brother` and `abort`, which
#           RP does not pronounce -- so this models his actual behaviour rather
#           than the accent he names.
import os
_BACKEND = os.environ.get("EN2THAANA_BACKEND", "rp")


def set_backend(name: str) -> None:
    global _BACKEND
    assert name in ("rp", "genam", "rhotic")
    _BACKEND = name


def _restore_r(word: str, ipa: str) -> str:
    """Re-insert /r/ that non-rhotic RP dropped but the spelling still shows.

    Walks the spelling and the phoneme string together; at each <r> in the
    spelling that has no /r/ nearby in the phonemes, inserts one after the
    current vowel. Deterministic and reversible."""
    ph = phonemes(ipa)
    n_r_spell = word.count("r")
    n_r_ipa = sum(1 for p in ph if p == "r")
    missing = n_r_spell - n_r_ipa
    if missing <= 0:
        return ipa
    out, seen = [], 0
    for i, p in enumerate(ph):
        out.append(p)
        nxt = ph[i + 1] if i + 1 < len(ph) else None
        # after a vowel, when the next phoneme is not already /r/
        if seen < missing and _is_v(p) and nxt != "r":
            out.append("r")
            seen += 1
    return "".join(out)


def _is_v(p: str) -> bool:
    return all(c in "\u026a\u025b\u00e6\u028c\u0252\u028a\u0259i\u0251\u0254\u025ceoau\u02d0"
               for c in p)


def lookup_backend(word: str) -> tuple[str | None, str]:
    w = word.strip().lower()
    if _BACKEND == "genam":
        cmu = load_cmudict()
        if w in cmu:
            return arpabet_to_ipa(cmu[w]), "cmudict_genam"
        ipa, src = lookup(w)
        return ipa, src
    ipa, src = lookup(w)
    if ipa and _BACKEND == "rhotic":
        return _restore_r(w, ipa), src
    return ipa, src


# ---------------------------------------------------------- diphthong expansion
# The wide (closing and centring) diphthongs are written across TWO Thaana
# slots -- the aibaifili construction:
#     brownie  ބްރައުނީ   =  ބ+ަ  އ+ު  ނ+ީ     /aU/ spans ަ and އު
# Expanding them into two vowel phonemes means the ordinary alifu-carrier rule
# ("a vowel with no preceding consonant opens an alifu") produces the second
# half by itself, with no diphthong special-case anywhere in the engine.
#
# The narrow diphthongs /eI/ and /@U/ are NOT expanded: they take a single long
# fili (ޭ, ޯ) in 91% and 85% of aligned gold words respectively. That asymmetry
# is measured -- see rules/vowels.tsv -- not stipulated.
EXPAND = {
    "a\u026a": ["a", "\u026a"],                 # aI  -> ަ + އި
    "a\u028a": ["a", "\u028a"],                 # aU  -> ަ + އު
    "\u0254\u026a": ["\u0254", "\u026a"],       # OI
    "\u026a\u0259": ["\u026a", "\u0259"],       # I@
    "e\u0259": ["e", "\u0259"],                 # e@
    "\u028a\u0259": ["\u028a", "\u0259"],       # U@
}


def expand_diphthongs(phones: list[str]) -> list[str]:
    out = []
    for p in phones:
        out.extend(EXPAND.get(p, [p]))
    return out


# --- tagged diphthong halves (supersedes the untagged EXPAND defined above) ---
# Each half is TAGGED with the diphthong it came from ("aI:1", "aI:2") rather
# than emitted as a plain /a/ or /I/. The first half of /aI/ is not the same
# thing as the monophthong /a/ and must not be counted into the same cell --
# leaving them untagged cost half a point by polluting both tables.
VOWEL_PHONE_CHARS = ("\u026a\u025b\u00e6\u028c\u0252\u028a\u0259i"
                     "\u0251\u0254\u025ceoau\u02d0")

WIDE = ["a\u026a", "a\u028a", "\u0254\u026a",
        "\u026a\u0259", "e\u0259", "\u028a\u0259"]
EXPAND = {d: [d + ":1", d + ":2"] for d in WIDE}


def is_vowel_phoneme(p: str) -> bool:
    """True for vowels, including the tagged halves of an expanded diphthong."""
    return all(c in VOWEL_PHONE_CHARS for c in p.split(":")[0])


def lookup_with_g2p(word: str) -> tuple[str | None, str]:
    """lookup_backend, then the OOV model as a last resort.

    Kept separate from lookup_backend so that induction and evaluation, which
    must measure the rule engine alone, cannot accidentally pull in model
    output. Only the application layer calls this."""
    ipa, src = lookup_backend(word)
    if ipa:
        return ipa, src
    try:
        from en2thaana.g2p import predict
    except Exception:
        return None, "none"
    guess = predict(word)
    return (guess, "g2p") if guess else (None, "none")
