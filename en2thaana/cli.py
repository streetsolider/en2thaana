"""Command line: `python -m en2thaana WORD [WORD...]` or `--build`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from en2thaana import transcribe
from en2thaana.ipa import load_britfone, load_cmudict, load_en_uk

ROOT = Path(__file__).resolve().parent.parent


def build(out: Path) -> None:
    """Bulk export over every word with a known pronunciation."""
    words = sorted(set(load_en_uk()) | set(load_britfone()) | set(load_cmudict()))
    words = [w for w in words if w.isalpha() and w.isascii()]
    n = bad = 0
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for w in words:
            r = transcribe(w)
            if not r.thaana:
                continue
            n += 1
            bad += (not r.legal)
            f.write(json.dumps({"word": r.word, "ipa": r.ipa, "thaana": r.thaana,
                                "source": r.source}, ensure_ascii=False) + "\n")
    print(f"wrote {out}  {n} entries  ({bad} failed a legality check)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="en2thaana", description=__doc__)
    ap.add_argument("words", nargs="*")
    ap.add_argument("--build", metavar="PATH", help="bulk export to JSONL")
    ap.add_argument("--explain", action="store_true",
                    help="show the IPA and the lexicon it came from")
    a = ap.parse_args(argv)

    if a.build:
        build(Path(a.build))
        return 0
    if not a.words:
        ap.print_help()
        return 1
    for w in a.words:
        r = transcribe(w)
        if not r.thaana:
            print(f"{w}\t-\tno pronunciation found")
            continue
        if a.explain:
            flag = "" if r.legal else "  ILLEGAL: " + "; ".join(r.problems)
            print(f"{r.word}\t/{r.ipa}/\t{r.thaana}\t[{r.source}]{flag}")
        else:
            print(f"{r.word}\t{r.thaana}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
