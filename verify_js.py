"""Check that the JavaScript rule engine agrees with the Python one.

web/public/engine.js is a hand port of en2thaana/render.py plus the parts of
orth.py and ipa.py it needs. A port drifts. Rather than hope it does not, both
are run over every word in the shipped dictionary and compared.

This is the gate that makes the port trustworthy: the browser only uses the JS
engine for words the model invents, which are exactly the words nobody has a
reference spelling for, so a silent divergence there would never be noticed by
looking at the output.

Needs node. Without it the check is skipped rather than failed, and says so.

    python verify_js.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web" / "public"
DIST = ROOT / "dist" / "en_thaana.jsonl"

CHECK_JS = """
import { convert } from "./engine.mjs";
import fs from "fs";
const lines = fs.readFileSync(process.argv[2], "utf8").split("\\n");
let n = 0, bad = 0; const samples = [];
for (const line of lines) {
  if (!line.trim()) continue;
  const r = JSON.parse(line); n++;
  const got = convert(r.ipa, r.word);
  if (got !== r.thaana) {
    bad++;
    if (samples.length < 10)
      samples.push(`${r.word}  /${r.ipa}/  python=${r.thaana}  js=${got}`);
  }
}
console.log(JSON.stringify({ n, bad, samples }));
"""


def main() -> int:
    if shutil.which("node") is None:
        print("node not found, skipping the JS/Python equivalence check")
        return 0
    if not DIST.exists():
        print("no dist/en_thaana.jsonl, skipping")
        return 0

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "engine.mjs").write_text(
            (WEB / "engine.js").read_text(encoding="utf-8")
            .replace('"./data.js"', '"./data.mjs"'), encoding="utf-8")
        (tmp / "data.mjs").write_text(
            (WEB / "data.js").read_text(encoding="utf-8"), encoding="utf-8")
        (tmp / "check.mjs").write_text(CHECK_JS, encoding="utf-8")

        r = subprocess.run(["node", str(tmp / "check.mjs"), str(DIST)],
                           capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        print("the JS engine failed to run:")
        print((r.stderr or "")[-1500:])
        return 1

    out = json.loads(r.stdout.strip().splitlines()[-1])
    print(f"compared {out['n']:,} words through both engines")
    if out["bad"] == 0:
        print("  identical on every one")
        return 0
    print(f"  {out['bad']} disagreements ({100*out['bad']/out['n']:.4f}%)")
    for s in out["samples"]:
        print("   " + s)
    return 1


if __name__ == "__main__":
    sys.exit(main())
