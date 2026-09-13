"""Thaana script mechanics: the letter/diacritic inventory and a word parser.

Thaana is an alphabet whose vowels are obligatory diacritics (fili) on the
consonant they follow. A consonant with no vowel carries sukun. This makes a
Thaana word parse cleanly into (letter, mark) units, which is what lets an IPA
string be aligned to a Thaana string without a learned aligner:

    unit                       phonemic content
    ------------------------   ----------------------------
    C + fili                   consonant + vowel
    C + sukun                  consonant alone (cluster/coda)
    alifu + fili               vowel alone (alifu is a null carrier)
    alifu + sukun              glottal stop / nothing

Reference: the Dhivehi Bahuge Academy's rules on sukun and hus noonu.
"""
from __future__ import annotations

ALIFU = "\u0787"
SUKUN = "\u07B0"

# The 24 base consonants, in code-point order.
CONSONANTS = "".join(chr(c) for c in range(0x0780, 0x0798))
# The 14 Arabic-derived "thikijehi" (dotted) letters. Only ޝ and ޜ are used for
# English -- they carry the two English sounds with no native Thaana letter.
DOTTED = "".join(chr(c) for c in range(0x0798, 0x07A6))
LETTERS = CONSONANTS + DOTTED

# The 10 fili, in code-point order, with their conventional Latin values.
FILI = {
    "\u07A6": "a",  "\u07A7": "aa", "\u07A8": "i",  "\u07A9": "ee",
    "\u07AA": "u",  "\u07AB": "oo", "\u07AC": "e",  "\u07AD": "ey",
    "\u07AE": "o",  "\u07AF": "oa",
}
MARKS = set(FILI) | {SUKUN}


def parse(word: str) -> list[tuple[str, str]] | None:
    """Thaana string -> [(letter, mark)]. mark is a fili, sukun, or '' when the
    letter carries nothing (which is not legal Thaana, but occurs in the gold
    data -- e.g. `ah` -> އާހ -- so it is represented rather than rejected).

    Returns None if the string contains a character that is not Thaana."""
    units: list[tuple[str, str]] = []
    for ch in word:
        if ch in LETTERS:
            units.append((ch, ""))
        elif ch in MARKS:
            if not units:
                return None
            letter, existing = units[-1]
            if existing:
                return None
            units[-1] = (letter, ch)
        else:
            return None
    return units


def is_vowel_slot(unit: tuple[str, str]) -> bool:
    """A unit whose phonemic content is a bare vowel: alifu carrying a fili."""
    return unit[0] == ALIFU and unit[1] in FILI


def render(units: list[tuple[str, str]]) -> str:
    return "".join(l + m for l, m in units)
