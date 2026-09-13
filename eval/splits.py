"""Stable train/dev/test assignment.

Uses sha1, NOT Python's hash(): hash() is randomised per process for strings, so
a split built on it silently changes between runs and every held-out number
measured against it is meaningless. Hashing the headword also means the split
survives regenerating the gold file and needs no seed stored anywhere.
"""
from __future__ import annotations

import hashlib


def split_of(word: str) -> str:
    h = int(hashlib.sha1(word.encode("utf-8")).hexdigest(), 16)
    r = h % 10
    return "train" if r < 8 else ("dev" if r == 8 else "test")
