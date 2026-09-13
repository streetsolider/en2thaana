"""Measure which phoneme source best predicts Hassan Hameed's spellings.

He names RP, but writes coda ރ that RP does not pronounce. Rather than argue
about which source is 'correct', induce the rule tables and score the engine
once per backend and report the three numbers.
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

for backend in ("rp", "genam", "rhotic"):
    for script in ("induce.py", "evaluate.py"):
        r = subprocess.run([PY, str(ROOT / "eval" / script)],
                           capture_output=True, text=True, encoding="utf-8",
                           env={**__import__("os").environ,
                                "EN2THAANA_BACKEND": backend,
                                "PYTHONIOENCODING": "utf-8"})
        if script == "evaluate.py":
            line = [l for l in r.stdout.splitlines() if "EXACT MATCH" in l]
            al = ""
            print(f"  {backend:<8} {line[0] if line else r.stderr[-300:]}")
