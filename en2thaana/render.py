"""Stage 2: IPA -> Thaana. Deterministic, no learned parameters.

The mapping tables are DATA (rules/*.tsv), induced from the gold pairs by
eval/induce.py, so every mapping carries the count of attested words supporting
it. Nothing is hardcoded here except the structural rules of the script itself.

THE STRUCTURAL RULES
--------------------
1. A consonant followed by a vowel takes that vowel's fili.
2. A consonant not followed by a vowel takes sukun -- EXCEPT ރ, which stands
   bare. Not a style choice: the gold has ރ bare 1,475 times against ރް twice,
   so writing ރް would be wrong in about one word in six. Read from
   rules/coda.tsv, where it is measured rather than asserted.
3. English clusters are written out. Dhivehi's own ban on initial clusters does
   NOT apply, because these are English words written by their English sound.
   `flotation` is ފްލޮޓޭޝަން, never އިފްލޮ...
4. A vowel with no preceding consonant takes alifu (އ) as a null carrier. This
   one rule produces both the word-initial carrier and the second half of every
   split diphthong, since those are expanded into two vowel slots.
5. The closing diphthongs /aI/ and /aU/ are written with two fili across an
   alifu -- the aibaifili construction:
       /aI/ -> ަ + އި      /aU/ -> ަ + އު
   The narrow diphthongs /eI/ and /@U/ do not: they take a single long fili
   (ޭ, ޯ). Measured, not assumed -- see rules/vowels.tsv.
6. A reduced vowel is resolved from the English LETTER that produced it when the
   spelling is available (en2thaana/orth.py). /@/ is written ޮ in `anthology`
   and ި in `anvil`; only the spelling distinguishes them.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from en2thaana import thaana as T
from en2thaana.ipa import phonemes, expand_diphthongs, is_vowel_phoneme
from en2thaana.orth import prepare

RULES = Path(__file__).resolve().parent.parent / "rules"

SPLIT_DIPHTHONGS = {
    "a\u026a": ("\u07A6", "\u07A8"),        # aI -> ަ + ި   (ައި)
    "a\u028a": ("\u07A6", "\u07AA"),        # aU -> ަ + ު   (ައު)
    "\u0254\u026a": ("\u07AE", "\u07A8"),   # OI -> ޮ + ި
    "\u026a\u0259": ("\u07A8", "\u07A6"),   # I@ -> ި + ަ
    "\u028a\u0259": ("\u07AB", "\u07A6"),   # U@ -> ޫ + ަ
}

VOWEL_CHARS = set("\u026a\u025b\u00e6\u028c\u0252\u028a\u0259i\u0251\u0254\u025ceoau")

# Legality invariants, from the Academy's published orthographic rules:
#  - a bare noonu is phonemic in Dhivehi (prenasalisation: ހިނގާ vs ހިންގާ) and
#    must never be emitted for an English word;
#  - a dot encodes an Arabic etymon, which an English word does not have, so the
#    only admissible dotted letters are the two carrying English sounds Thaana
#    has no plain letter for: ޝ /S/ and ޜ /Z/;
#  - ޅ and ޏ are Dhivehi-only and appear zero times in 8,662 gold words.
ALLOWED_DOTTED = {"\u079D", "\u079C"}
FORBIDDEN = {"\u0785", "\u078F"}

MIN_ORTH_EVIDENCE = 3       # a cell resting on one or two words is noise
MIN_ORTH_SHARE = 0.50       # and a coin-flip cell is not evidence either

# The /juː/ glide. Three components, two settled by the gold data and one
# settled by a speaker:
#
#   ި  on the preceding consonant   218 words vs 9 using sukun   (gold)
#   ު  short, not long ޫ            193 words vs 22              (gold)
#   ޔ  as the carrier               a native speaker's judgement (see below)
#
# The carrier is the one genuinely contested cell in the engine. Hassan Hameed
# splits 124 alifu to 94 yaviyani, and conditioning it on the English spelling
# finds nothing to condition on: <u> is 220 of the 250 cases and splits 113/107.
#
# So the gold cannot decide it, and a 57/43 split from a single lexicographer is
# not evidence of a convention -- it is evidence that he had no convention. A
# native speaker judged ޔ more accurate, and on a question the data leaves open
# that is the better authority. This costs measured agreement with Hassan
# Hameed, because the majority of his spellings use alifu; it is recorded here
# so the drop is never mistaken for a regression.
GLIDE_FILI = "ި"            # ި
GLIDE_CARRIER = "ޔ"         # ޔ  -- speaker's choice over HH's 57% alifu
SHORTEN = {"ޫ": "ު"}   # ޫ -> ު after the glide


def is_vowel(p: str) -> bool:
    return is_vowel_phoneme(p)


def _read(name, key, val):
    out = {}
    with (RULES / name).open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            out[r[key]] = r[val]
    return out


@lru_cache(maxsize=1)
def _tables():
    # Phoneme-only shares, needed to decide whether an orthographic cell has
    # earned the right to override one.
    phon_share = {}
    with (RULES / "vowels.tsv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            phon_share[r["phoneme"]] = float(r["share"])

    orth = {}
    with (RULES / "vowels_orth.tsv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if int(r["total"]) < MIN_ORTH_EVIDENCE:
                continue
            share = float(r["share"])
            # An orthographic cell replaces the phoneme-only rule only when it
            # is MORE confident than the rule it displaces. /ɒ/ spelled <a> is
            # 33% (7 of 21) and was overriding the 84% ޮ, which is how `was`
            # came out ވާޒް instead of ވޮޒް. A feature that adds information
            # must not be allowed to subtract certainty.
            if share < max(MIN_ORTH_SHARE, phon_share.get(r["phoneme"], 0.0)):
                continue
            orth[(r["phoneme"], r["orth"])] = r["fili"]
    return (_read("consonants.tsv", "phoneme", "thaana"),
            _read("vowels.tsv", "phoneme", "fili"),
            _read("coda.tsv", "letter", "coda_form"),
            orth)


def check(word: str) -> list[str]:
    """Legality violations in a rendered Thaana word. Empty means legal."""
    bad = []
    units = T.parse(word)
    if units is None:
        return ["not parseable as Thaana"]
    for letter, mark in units:
        if letter in FORBIDDEN:
            bad.append(letter + " is Dhivehi-only, never used for English")
        if letter in T.DOTTED and letter not in ALLOWED_DOTTED:
            bad.append(letter + " is a dotted letter with no English use")
        if letter == "\u0782" and mark == "":
            bad.append("bare noonu: prenasalisation is phonemic (s16)")
    return bad


def convert(ipa: str, word: str | None = None) -> str:
    """IPA -> Thaana.

    `word` is the English spelling. Given it, reduced vowels are resolved from
    the letter that produced them; without it the engine falls back to the
    phoneme-only table and loses the /@/ distinctions."""
    cons, vows, coda, orth = _tables()
    ph_all, hints = prepare(word or "", phonemes(ipa)) if word else (
        expand_diphthongs(phonemes(ipa)), None)
    vi = 0
    units = []

    def put_vowel(fili):
        """Attach a fili to a pending bare consonant, else open an alifu."""
        if units and units[-1][1] == "":
            units[-1][1] = fili
        else:
            units.append([T.ALIFU, fili])

    i = 0
    while i < len(ph_all):
        p = ph_all[i]

        # /j/ between a consonant and a vowel is not written as a consonant.
        # It becomes an i-fili on the letter before it, and the vowel that
        # follows opens its own carrier: computer is \u0786\u07AE\u0789\u07B0\u0795\u07A8\u0787\u07AA\u0793\u07A6\u0783, not \u0786\u07AE\u0789\u07B0\u0795\u07B0\u0794\u07AB\u0793\u07A6\u0783.
        # Measured over the gold: the preceding consonant takes \u07A8 in 218 words
        # against 9 that use sukun, and the vowel is the SHORT \u07AA in 193 against
        # 22 long. See GLIDE_* below for the carrier choice.
        if (p == "j" and i + 1 < len(ph_all) and is_vowel(ph_all[i + 1])
                and units and units[-1][1] == ""):
            units[-1][1] = GLIDE_FILI
            nxt = ph_all[i + 1]
            hint = hints[vi] if hints is not None else None
            vi += 1
            fili = (orth.get((nxt, hint)) if hint else None) or vows.get(nxt, "\u07A6")
            units.append([GLIDE_CARRIER, SHORTEN.get(fili, fili)])
            i += 2
            continue

        if is_vowel(p):
            hint = hints[vi] if hints is not None else None
            vi += 1
            if p in SPLIT_DIPHTHONGS:
                a, b = SPLIT_DIPHTHONGS[p]
                put_vowel(a)
                units.append([T.ALIFU, b])
            else:
                fili = orth.get((p, hint)) if hint else None
                put_vowel(fili or vows.get(p, "\u07A6"))
        else:
            units.append([cons.get(p, ""), ""])
        i += 1

    for u in units:
        if u[1] == "":
            u[1] = "" if coda.get(u[0]) == "BARE" else T.SUKUN
    return "".join(l + m for l, m in units if l)
