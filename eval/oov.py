"""What the OOV model actually costs, end to end.

The model's own word accuracy is not the number that matters -- what matters is
how often the FINAL Thaana string is the same as it would have been if the word
had been in a dictionary. A phoneme error that the rule engine maps onto the
same grapheme is harmless; one that changes a fili is not.

So: take words the model was never trained on, transcribe each one twice --
once through the lexicon (the reference path) and once through the model -- and
compare the Thaana. That isolates the model's contribution from the engine's
own ~48% agreement with Hassan Hameed, which is a separate measurement.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from en2thaana.g2p import available, predict, status
from en2thaana.ipa import lookup_backend
from en2thaana.render import convert

N = 4000


def bucket(w: str) -> int:
    return int(hashlib.sha1(("g2p" + w).encode()).hexdigest(), 16) % 40


def main() -> None:
    if not available():
        print("no model at models/g2p.pt")
        return
    from en2thaana.ipa import load_britfone, load_cmudict, load_en_uk

    # Bucket 39 is the model's own held-out test split -- never trained on.
    words = [w for w in sorted(set(load_en_uk()) | set(load_britfone())
                               | set(load_cmudict()))
             if w.isalpha() and w.isascii() and 2 < len(w) <= 26 and bucket(w) == 39]
    words = words[:N]

    same = n = 0
    phon_same = 0
    misses = []
    for w in words:
        ref_ipa, src = lookup_backend(w)
        if not ref_ipa:
            continue
        got_ipa = predict(w)
        if not got_ipa:
            continue
        n += 1
        phon_same += (got_ipa == ref_ipa)
        a, b = convert(ref_ipa, w), convert(got_ipa, w)
        if a == b:
            same += 1
        elif len(misses) < 25:
            misses.append((w, ref_ipa, got_ipa, a, b))

    print(f"model: {status()}")
    print(f"held-out words scored: {n}\n")
    print(f"  identical IPA            {phon_same/n:7.2%}")
    print(f"  identical THAANA         {same/n:7.2%}   <- the number that matters")
    print("\nthe Thaana agrees more often than the IPA does: the rule engine "
          "\nmaps several phoneme confusions onto the same grapheme.\n")
    print("sample disagreements (word | reference ipa | model ipa | ref | model):")
    for w, ri, gi, a, b in misses[:12]:
        print(f"  {w:<16} /{ri}/ vs /{gi}/   {a}  vs  {b}")


if __name__ == "__main__":
    main()
