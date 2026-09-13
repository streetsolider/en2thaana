"""Out-of-vocabulary fallback: predict RP phonemes for a word no lexicon has.

This is the only learned component in the system, and it is reached only after
every lookup has failed. It emits PHONEMES, never Thaana -- the deterministic
rule engine converts its output exactly as it converts a dictionary word, so no
decision that a rule makes is ever delegated to a model.

torch is an optional dependency. Import failures and a missing checkpoint are
both non-fatal: the pipeline falls back to reporting the word as unknown, which
is the honest outcome and the one the package had before the model existed.
Results carry source="g2p" so the model's share of any output is measurable.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

CKPT = Path(__file__).resolve().parent.parent / "models" / "g2p.pt"
MAX_SRC = 26


@lru_cache(maxsize=1)
def _load():
    """-> (model, src_stoi, tgt_itos, device) or None if unavailable."""
    if not CKPT.exists():
        return None
    try:
        import torch
        from train_g2p import G2P
    except Exception:
        return None
    try:
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        src_itos, tgt_itos = ck["src_itos"], ck["tgt_itos"]
        model = G2P(len(src_itos), len(tgt_itos))
        model.load_state_dict(ck["model"])
        model.eval()
        return model, {c: i for i, c in enumerate(src_itos)}, tgt_itos, "cpu"
    except Exception:
        return None


def available() -> bool:
    return _load() is not None


def status() -> str | None:
    m = _load()
    if m is None:
        return None
    n = sum(p.numel() for p in m[0].parameters())
    return f"{n/1e6:.1f}M parameters"


@lru_cache(maxsize=8192)
def predict(word: str) -> str | None:
    """English word -> RP IPA string, or None when the model is unavailable."""
    loaded = _load()
    if loaded is None:
        return None
    model, stoi, itos, dev = loaded
    w = word.strip().lower()
    if not w.isalpha() or not w.isascii() or len(w) > MAX_SRC:
        return None
    if any(c not in stoi for c in w):
        return None

    import torch
    src = torch.tensor([[stoi[c] for c in w]], device=dev)
    with torch.no_grad():
        ys = model.greedy(src)[0].tolist()
    out = []
    for t in ys[1:]:
        if t in (0, 1):          # PAD, BOS
            continue
        if t == 2:               # EOS
            break
        out.append(itos[t])
    return "".join(out) or None
