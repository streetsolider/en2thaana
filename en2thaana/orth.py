"""The orthographic feature: which English LETTER produced a sound.

WHY THIS EXISTS
---------------
Hassan Hameed's renderings are not a pure phonemic transcription. He spells from
the English spelling wherever the sound alone would be ambiguous. Measured over
the gold pairs, words whose single schwa is written:

    ޮ  absolute acrimony agony alcohol alimony allegory anthology   <- spelled <o>
    ާ  algebra alpha ambrosia ammonia amnesia angora antenna area    <- final <a>
    ި  anvil april capricorn caret chervil civil clarinet            <- spelled <i>
    ު  awful bagful bashful deceitful doleful dreadful fitful        <- spelled <u>

All four are the same phoneme /@/. A pipeline that goes English -> IPA -> Thaana
has thrown the distinguishing information away one step before it is needed.

The same is true of /r/. He names Received Pronunciation, which is non-rhotic,
but writes ރ in `acre`, `adder`, `adverb`, `afternoon` -- 600 gold words carry a
ރ that RP does not pronounce. He is writing the letter, not the sound.

So the engine is given the spelling as a second input. This is declared openly
and its contribution is measured (eval/backends.py) rather than hidden behind a
claim that the mapping is purely phonemic.

HOW THE CORRESPONDENCE IS FOUND
-------------------------------
No aligner. Take the maximal runs of vowel letters in the spelling and the vowel
phonemes in order; when the two counts are equal the correspondence is
positional and unambiguous. When they are not, no feature is offered and the
engine falls back to the phoneme-only table. The share of words that qualify is
reported, so the fallback rate is never a guess.
"""
from __future__ import annotations

import re

VOWEL_LETTERS = "aeiouy"
_RUNS = re.compile(f"[{VOWEL_LETTERS}]+")

VOWEL_PHONE_CHARS = "\u026a\u025b\u00e6\u028c\u0252\u028a\u0259i\u0251\u0254\u025ceoau\u02d0"


def _is_v(p: str) -> bool:
    return all(c in VOWEL_PHONE_CHARS for c in p)


def spelling_form(word: str) -> str:
    """The spelling with silent final <e> removed.

    English marks vowel length with a final <e> that is not itself pronounced.
    Counting it as a vowel run makes the run count exceed the vowel-phoneme
    count, which silently disables the orthographic feature for a very large
    class of words -- abase, abate, ablaze, abode all failed on this alone.
    Dropping it recovers them."""
    w = word.lower()
    if len(w) > 3 and w.endswith("e") and w[-2] not in VOWEL_LETTERS:
        return w[:-1]
    return w


def vowel_spans(word: str) -> list[tuple[int, int, str]]:
    """(start, end, text) for each maximal vowel-letter run, over the spelling
    with silent final <e> removed."""
    return [(m.start(), m.end(), m.group())
            for m in _RUNS.finditer(spelling_form(word))]


def vowel_groups(word: str) -> list[str]:
    return [t for _s, _e, t in vowel_spans(word)]


def orth_hints(word: str, n_vowel_phonemes: int) -> list[str] | None:
    """One orthographic hint per vowel phoneme, or None when the counts disagree.

    The hint is the vowel-letter run, marked '^' when it starts the word and '#'
    when it ends it. Both positions behave differently and both distinctions pay
    for themselves: word-initial unstressed <a> is written ެ (aback, abate,
    abduct) where medial <a> is ަ, and word-final <a> is ާ (algebra, antenna)."""
    spans = vowel_spans(word)
    if len(spans) != n_vowel_phonemes:
        return None
    w = spelling_form(word)
    out = []
    for start, end, text in spans:
        mark = ("^" if start == 0 else "") + ("#" if end == len(w) else "")
        out.append(mark + text)
    return out


def restore_r(word: str, phones: list[str]) -> list[str]:
    """Re-insert the /r/ that non-rhotic RP drops but the spelling still shows.

    Placed by POSITION, not by count: a vowel run immediately followed by <r> in
    the spelling gets an /r/ after its phoneme, unless one is already there. An
    earlier version inserted after any vowel until the counts matched and cost
    two points of accuracy -- position is what makes this work."""
    spans = vowel_spans(word)
    vidx = [i for i, p in enumerate(phones) if _is_v(p)]
    if len(spans) != len(vidx):
        return phones
    w = spelling_form(word)
    insert_after = set()
    for (start, end, _t), pi in zip(spans, vidx):
        if w[end:end + 1] == "r":
            nxt = phones[pi + 1] if pi + 1 < len(phones) else None
            if nxt != "r":
                insert_after.add(pi)
    if not insert_after:
        return phones
    out = []
    for i, p in enumerate(phones):
        out.append(p)
        if i in insert_after:
            out.append("r")
    return out


def prepare(word: str, phones: list[str]):
    """-> (expanded phones, one hint per vowel slot, or None)

    The order matters. Hints are matched against the UNEXPANDED phonemes, since
    a spelling has one vowel run where /aI/ has two halves -- `bite` is b-i-t-e,
    one run, but two vowel slots. Matching after expansion made every diphthong
    word fail the count check and silently lose its hint. So: match first, then
    expand both lists in step, with both halves of a diphthong inheriting the
    hint of the letter that produced them."""
    from en2thaana.ipa import EXPAND, is_vowel_phoneme

    phones = drop_epenthetic(word, phones)
    n_v = sum(1 for p in phones if is_vowel_phoneme(p))
    hints = orth_hints(word, n_v)
    phones = restore_r(word, phones)

    out_p, out_h, vi = [], [], 0
    for p in phones:
        if not is_vowel_phoneme(p):
            out_p.append(p)
            continue
        h = hints[vi] if hints is not None else None
        vi += 1
        parts = EXPAND.get(p, [p])
        out_p.extend(parts)
        out_h.extend([h] * len(parts))
    return out_p, (out_h if hints is not None else None)


SONORANTS = {"l", "n", "m", "r"}


def drop_epenthetic(word: str, phones: list[str]) -> list[str]:
    """Delete schwas that no letter in the spelling accounts for.

    English inserts a schwa before a syllabic sonorant that the spelling does not
    write: `able` is a-b-l, one vowel letter, but /eIb@l/ has two vowel sounds.
    Hassan Hameed follows the spelling and writes no fili there -- އޭބްލް, with
    sukun on the ބ -- and the same holds for -al, -el, -on, -en, -ism.

    So when the phonemes carry MORE vowels than the spelling has vowel runs, the
    surplus are epenthetic. They are removed, preferring schwas that sit before a
    sonorant and after a consonant, rightmost first. If that cannot reconcile the
    two counts the phones are returned untouched and the word simply gets no
    orthographic hints -- a guess here would be worse than a fallback."""
    from en2thaana.ipa import is_vowel_phoneme

    n_runs = len(vowel_spans(word))
    idx = [i for i, p in enumerate(phones) if is_vowel_phoneme(p)]
    surplus = len(idx) - n_runs
    if surplus <= 0:
        return phones

    cands = []
    for i in idx:
        if phones[i] != "\u0259":
            continue
        prev = phones[i - 1] if i else None
        nxt = phones[i + 1] if i + 1 < len(phones) else None
        if prev is not None and not is_vowel_phoneme(prev) and nxt in SONORANTS:
            cands.append(i)
    if len(cands) < surplus:
        return phones
    drop = set(cands[-surplus:])
    return [p for i, p in enumerate(phones) if i not in drop]
