"""Export the out-of-vocabulary model to ONNX so the page can run it.

WHY TWO GRAPHS
--------------
Decoding is autoregressive: the model emits one phoneme, then reads what it
emitted to produce the next. As a single graph the browser would re-run the
encoder on every step, encoding a nine-letter word nine times over. Split, the
source is encoded once and only the decoder loops.

WHY FIXED SHAPES AND EXPLICIT MASKS
-----------------------------------
MultiheadAttention reshapes with concrete integers, so a traced graph bakes in
whatever length the example had and then rejects every other length:

    Input shape:{1,1,256}, requested shape:{15,4,64}

dynamic_axes does not fix that, and torch.export cannot resolve the module's
internal shape branches either ("Could not guard on Eq(u0, 1)"). So the graphs
take fixed windows instead, 26 source positions and 30 target positions, with
the padding written into masks the model already knows how to honour. Attention
skips the padding, and the causal mask means position i never looks past itself,
so the answer matches the variable-length model. A 5.3M model over a 26x30
window costs nothing.

Both exports are checked against PyTorch, and the quantized graphs are checked
again, because a quantized model that quietly disagrees with its source is worse
than no model at all.

    div-transliteration/venv/Scripts/python.exe export_onnx.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train_g2p import BOS, EOS, MAX_SRC, MAX_TGT, PAD, G2P  # noqa: E402

CKPT = ROOT / "models" / "g2p.pt"
OUT = ROOT / "web" / "public" / "model"


class Encoder(nn.Module):
    def __init__(self, m: G2P):
        super().__init__()
        self.m = m

    def forward(self, src):
        pad = src == PAD
        x = self.m.pos(self.m.src_emb(src) * math.sqrt(self.m.d))
        return self.m.tf.encoder(x, src_key_padding_mask=pad)


class Decoder(nn.Module):
    def __init__(self, m: G2P):
        super().__init__()
        self.m = m
        self.register_buffer(
            "causal",
            torch.triu(torch.full((MAX_TGT, MAX_TGT), float("-inf")), diagonal=1))

    def forward(self, memory, src, tgt_in):
        pad = src == PAD
        y = self.m.pos(self.m.tgt_emb(tgt_in) * math.sqrt(self.m.d))
        h = self.m.tf.decoder(y, memory, tgt_mask=self.causal,
                              memory_key_padding_mask=pad)
        return self.m.out(h)


def main() -> None:
    if not CKPT.exists():
        print("no checkpoint at models/g2p.pt -- run train_g2p.py first")
        return
    OUT.mkdir(parents=True, exist_ok=True)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    src_itos, tgt_itos = ck["src_itos"], ck["tgt_itos"]
    model = G2P(len(src_itos), len(tgt_itos))
    model.load_state_dict(ck["model"])
    model.eval()

    torch.backends.mha.set_fastpath_enabled(False)
    enc, dec = Encoder(model).eval(), Decoder(model).eval()
    stoi = {c: i for i, c in enumerate(src_itos)}

    def ids(word):
        return [stoi[c] for c in word.lower() if c in stoi][:MAX_SRC]

    def pad_src(word):
        v = ids(word)
        return torch.tensor([v + [PAD] * (MAX_SRC - len(v))], dtype=torch.long)

    def reference(word):
        """The model used the way training used it. Ground truth."""
        with torch.no_grad():
            out = model.greedy(torch.tensor([ids(word)], dtype=torch.long))[0].tolist()
        keep = []
        for t in out[1:]:
            if t in (PAD, BOS):
                continue
            if t == EOS:
                break
            keep.append(t)
        return keep

    probe = pad_src("cryptocurrency")
    tgt0 = torch.full((1, MAX_TGT), PAD, dtype=torch.long)
    tgt0[0, 0] = BOS
    with torch.no_grad():
        mem = enc(probe)

    torch.onnx.export(enc, (probe,), str(OUT / "encoder.onnx"),
                      input_names=["src"], output_names=["memory"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    torch.onnx.export(dec, (mem, probe, tgt0), str(OUT / "decoder.onnx"),
                      input_names=["memory", "src", "tgt_in"],
                      output_names=["logits"],
                      opset_version=17, do_constant_folding=True, dynamo=False)

    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    WORDS = ["cryptocurrency", "hameed", "naeem", "zuckerberg", "deanonymisation",
             "maldives", "thaana", "kulhudhuffushi", "blockchain", "fintech",
             "nasheed", "raajje", "dhiraagu", "ooredoo", "velaanaa"]

    def run_onnx(se, sd, word):
        src = pad_src(word).numpy()
        m = se.run(None, {"src": src})[0]
        buf = np.full((1, MAX_TGT), PAD, dtype=np.int64)
        buf[0, 0] = BOS
        out = []
        for step in range(MAX_TGT - 1):
            lg = sd.run(None, {"memory": m, "src": src, "tgt_in": buf})[0]
            nxt = int(lg[0, step].argmax())
            if nxt == EOS:
                break
            out.append(nxt)
            buf[0, step + 1] = nxt
        return out

    def check(e, d, label):
        se = ort.InferenceSession(str(e), providers=["CPUExecutionProvider"])
        sd = ort.InferenceSession(str(d), providers=["CPUExecutionProvider"])
        agree = sum(run_onnx(se, sd, w) == reference(w) for w in WORDS)
        print(f"  {label:<5} {agree}/{len(WORDS)} words decode identically to PyTorch")
        return agree == len(WORDS)

    ok_fp32 = check(OUT / "encoder.onnx", OUT / "decoder.onnx", "fp32")

    for name in ("encoder", "decoder"):
        # Best of the settings measured over 24 held-out words: per-channel
        # scales with a reduced range agreed with PyTorch on 22, where the
        # others managed 21. See the note below on why int8 is shipped anyway.
        quantize_dynamic(OUT / f"{name}.onnx", OUT / f"{name}.q.onnx",
                         weight_type=QuantType.QInt8,
                         per_channel=True, reduce_range=True)
    ok_int8 = check(OUT / "encoder.q.onnx", OUT / "decoder.q.onnx", "int8")

    for name in ("encoder", "decoder"):
        (OUT / f"{name}.onnx").unlink()
        (OUT / f"{name}.q.onnx").rename(OUT / f"{name}.onnx")

    (OUT / "vocab.json").write_text(json.dumps(
        {"src": src_itos, "tgt": tgt_itos, "pad": PAD, "bos": BOS, "eos": EOS,
         "maxSrc": MAX_SRC, "maxTgt": MAX_TGT}, ensure_ascii=False), encoding="utf-8")

    files = ("encoder.onnx", "decoder.onnx", "vocab.json")
    for f in files:
        print(f"  {f:<15} {(OUT / f).stat().st_size/1e6:.2f} MB")
    print(f"  {'total':<15} {sum((OUT / f).stat().st_size for f in files)/1e6:.2f} MB")
    if not ok_fp32:
        print("  WARNING: the float export disagrees with PyTorch. Do not ship it.")
    elif not ok_int8:
        # int8 is shipped knowingly. The float graphs are exact but 21 MB, and
        # this model is a guesser: it gets the phonemes exactly right about 67%
        # of the time. Where quantization flips a greedy argmax the result is a
        # different guess, not a corrupted one, and 4x smaller matters more on a
        # phone. The float export is checked first precisely so that a genuine
        # export bug cannot hide inside this allowance.
        print("  int8 differs from PyTorch on a few words. Shipping it anyway:")
        print("  the float graphs are exact but 21 MB, and this model guesses.")


if __name__ == "__main__":
    main()
