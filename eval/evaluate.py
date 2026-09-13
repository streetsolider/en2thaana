"""Exact-match accuracy against the held-out gold pairs, with error attribution.

Reports which phoneme is responsible for each mismatch, so a failure points at a
rule rather than at a vibe. Also emits hh_disagreements.tsv: every gold entry the
engine disagrees with, for a person to adjudicate -- never a silent overwrite.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from en2thaana import thaana as T
from en2thaana.ipa import lookup_backend as lookup, phonemes
from en2thaana.render import convert, is_vowel
from eval.gold import load_gold
from eval.splits import split_of

REPORTS = Path(__file__).resolve().parent / "reports"


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main(split: str = "all") -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    gold = load_gold()
    if split != "all":
        gold = [(e, t) for e, t in gold if split_of(e) == split]

    n = ok = 0
    dist = collections.Counter()
    blame = collections.Counter()
    rows = []
    for en, th in gold:
        ipa, src = lookup(en)
        if not ipa:
            continue
        n += 1
        got = convert(ipa, en)
        if got == th:
            ok += 1
            continue
        d = edit_distance(got, th)
        dist[min(d, 5)] += 1
        # attribute: which phonemes' graphemes are absent from the gold string
        for p in phonemes(ipa):
            g = convert(p) if is_vowel(p) else convert(p)
            if g and g[0] not in th:
                blame[p] += 1
        rows.append((en, ipa, th, got, str(d), src))

    print(f"split={split}  scored={n}")
    print(f"EXACT MATCH  {ok}/{n} = {100*ok/n:.2f}%")
    print("\nedit distance of the misses:")
    for d, c in sorted(dist.items()):
        label = f"{d}" if d < 5 else "5+"
        print(f"  {label:>2}  {c:>5}  {100*c/max(n-ok,1):5.1f}%")
    print("\ntop phonemes implicated in misses:")
    for p, c in blame.most_common(12):
        print(f"  {p:<4} {c:>5}")

    out = REPORTS / "hh_disagreements.tsv"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write("english\tipa\thassanhameed\tengine\tedit_distance\tipa_source\n")
        for r in sorted(rows, key=lambda x: int(x[4])):
            f.write("\t".join(r) + "\n")
    print(f"\nwrote {out} ({len(rows)} disagreements)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
