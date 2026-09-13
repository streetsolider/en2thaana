"""Induce the phoneme -> Thaana mapping from the gold pairs, by alignment.

WHY INDUCE RATHER THAN ASSERT
-----------------------------
A hand-written mapping table is an opinion. This derives the table from the 8.6k
attested English->Thaana spellings, so every row carries the count of gold words
supporting it and a reviewer can see which mappings rest on a single example.

HOW THE ALIGNMENT WORKS
-----------------------
No learned aligner is needed. A Thaana word parses into (letter, mark) units
whose phonemic content is fully determined (en2thaana/thaana.py):

    alifu + fili  -> one vowel slot
    C     + fili  -> one consonant slot, then one vowel slot
    C     + sukun -> one consonant slot
    C     + none  -> one consonant slot

That yields typed slots (C or V); the IPA side yields typed phonemes. Where the
two type-sequences are IDENTICAL the alignment is unambiguous and each
(phoneme, grapheme) pair is counted directly. Pairs whose patterns disagree are
not guessed at -- they go to reports/unaligned.tsv and carry no weight.

THREE TABLES COME OUT
---------------------
rules/consonants.tsv    phoneme -> letter
rules/vowels.tsv        phoneme -> fili                      (phoneme only)
rules/vowels_orth.tsv   (phoneme, English vowel letter) -> fili
rules/coda.tsv          letter -> SUKUN | BARE

coda.tsv is measured over EVERY gold word, aligned or not: whether a letter
takes sukun or stands bare is visible in the Thaana string alone.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from en2thaana import thaana as T
from en2thaana.ipa import lookup_backend as lookup, phonemes, expand_diphthongs, is_vowel_phoneme
from en2thaana.orth import prepare
from eval.gold import load_gold
from eval.splits import split_of

ROOT = Path(__file__).resolve().parent.parent
RULES = ROOT / "rules"
REPORTS = ROOT / "eval" / "reports"

VOWEL_CHARS = set("\u026a\u025b\u00e6\u028c\u0252\u028a\u0259i\u0251\u0254\u025ceoau")

TAB = "\t"
NL = "\n"


def is_vowel(p: str) -> bool:
    return is_vowel_phoneme(p)


def slots(units):
    out = []
    for letter, mark in units:
        if T.is_vowel_slot((letter, mark)):
            out.append(("V", mark))
            continue
        out.append(("C", letter))
        if mark in T.FILI:
            out.append(("V", mark))
    return out


def main() -> None:
    RULES.mkdir(exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    gold = load_gold()

    coda = collections.defaultdict(collections.Counter)
    for _en, th in gold:
        if split_of(_en) != "train":
            continue
        units = T.parse(th)
        if units is None:
            continue
        for letter, mark in units:
            if mark == T.SUKUN:
                coda[letter]["sukun"] += 1
            elif mark == "":
                coda[letter]["bare"] += 1

    cons = collections.defaultdict(collections.Counter)
    vows = collections.defaultdict(collections.Counter)
    vows_orth = collections.defaultdict(collections.Counter)
    stat = collections.Counter()
    unaligned = []

    for en, th in gold:
        # Induce on train only: a table fitted on the test words would make the
        # held-out number a measurement of memorisation.
        if split_of(en) != "train":
            continue
        stat["gold"] += 1
        ipa, _src = lookup(en)
        if not ipa:
            stat["no ipa"] += 1
            continue
        units = T.parse(th)
        if units is None:
            stat["thaana unparseable"] += 1
            continue
        ph, hints = prepare(en, phonemes(ipa))
        sl = slots(units)
        ptypes = ["V" if is_vowel(p) else "C" for p in ph]
        stypes = [t for t, _ in sl]
        if ptypes != stypes:
            stat["pattern mismatch"] += 1
            unaligned.append((en, ipa, th, "".join(ptypes), "".join(stypes)))
            continue
        stat["aligned"] += 1
        vi = 0
        for p, (t, g) in zip(ph, sl):
            (vows if t == "V" else cons)[p][g] += 1
            if t == "V":
                if hints is not None:
                    vows_orth[(p, hints[vi])][g] += 1
                vi += 1

    def write(path, table, header):
        with path.open("w", encoding="utf-8", newline=NL) as f:
            f.write("phoneme" + TAB + header + TAB + "count" + TAB + "share"
                    + TAB + "total" + TAB + "alternatives" + NL)
            for p in sorted(table, key=lambda x: -sum(table[x].values())):
                c = table[p]
                tot = sum(c.values())
                top, n = c.most_common(1)[0]
                alts = " ".join(g + "(" + str(k) + ")" for g, k in c.most_common()[1:6])
                f.write(TAB.join([p, top, str(n), f"{n/tot:.4f}", str(tot), alts]) + NL)

    write(RULES / "consonants.tsv", cons, "thaana")
    write(RULES / "vowels.tsv", vows, "fili")

    # The feature that separates anthology's /@/ (spelled <o>, written ޮ) from
    # anvil's (spelled <i>, written ި). Same phoneme, different letter.
    with (RULES / "vowels_orth.tsv").open("w", encoding="utf-8", newline=NL) as f:
        f.write(TAB.join(["phoneme", "orth", "fili", "count", "share", "total"]) + NL)
        for key in sorted(vows_orth, key=lambda k: -sum(vows_orth[k].values())):
            p, o = key
            c = vows_orth[key]
            tot = sum(c.values())
            top, n = c.most_common(1)[0]
            f.write(TAB.join([p, o, top, str(n), f"{n/tot:.4f}", str(tot)]) + NL)

    # Thaana requires sukun on a vowelless consonant, with one attested
    # exception: ރ stands alone (1,475 bare vs 2 sukun in the gold).
    with (RULES / "coda.tsv").open("w", encoding="utf-8", newline=NL) as f:
        f.write(TAB.join(["letter", "coda_form", "bare", "sukun", "bare_share"]) + NL)
        for l in sorted(coda, key=lambda x: -sum(coda[x].values())):
            c = coda[l]
            tot = c["bare"] + c["sukun"]
            share = c["bare"] / tot if tot else 0.0
            form = "BARE" if (share > 0.5 and tot >= 20) else "SUKUN"
            f.write(TAB.join([l, form, str(c["bare"]), str(c["sukun"]),
                              f"{share:.4f}"]) + NL)

    with (REPORTS / "unaligned.tsv").open("w", encoding="utf-8", newline=NL) as f:
        f.write(TAB.join(["english", "ipa", "thaana", "phoneme_pattern",
                          "slot_pattern"]) + NL)
        for row in unaligned:
            f.write(TAB.join(row) + NL)

    tot = stat["gold"]
    print(f"gold pairs           {tot}")
    for k in ["aligned", "pattern mismatch", "no ipa", "thaana unparseable"]:
        print(f"  {k:<20} {stat[k]:>6}  {100*stat[k]/tot:5.1f}%")
    bare = [l for l in coda
            if coda[l]["bare"] > coda[l]["sukun"] and sum(coda[l].values()) >= 20]
    print(f"\nwrote consonants.tsv ({len(cons)}), vowels.tsv ({len(vows)}), "
          f"vowels_orth.tsv ({len(vows_orth)} cells), coda.tsv")
    print(f"letters written BARE in coda: {' '.join(bare) or '(none)'}")


if __name__ == "__main__":
    main()
