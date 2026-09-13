"""en2thaana -- English pronunciation written in Thaana script.

    >>> from en2thaana import transcribe
    >>> transcribe("decision").thaana
    'ޑިސިޜަން'

Two stages. Stage 1 looks the word up in an RP pronunciation lexicon (ipa-dict
en_UK, Britfone, or CMUdict with a General-American-to-RP transform). Stage 2 is
a deterministic rule engine whose tables are induced from attested spellings and
live in rules/*.tsv, each row carrying the number of words that support it.

No learned parameters are involved in Stage 2. See README.md for why, and for
the measured accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass

from en2thaana.ipa import lookup_backend, phonemes
from en2thaana.render import check, convert

__all__ = ["transcribe", "Result"]


@dataclass
class Result:
    word: str
    ipa: str | None
    thaana: str | None
    source: str
    legal: bool
    problems: list


def transcribe(word: str) -> Result:
    ipa, source = lookup_backend(word)
    if not ipa:
        return Result(word, None, None, "none", False, ["no pronunciation found"])
    th = convert(ipa, word)
    problems = check(th)
    return Result(word, ipa, th, source, not problems, problems)
