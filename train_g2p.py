"""Train the out-of-vocabulary G2P model: English letters -> RP phonemes.

WHY A MODEL HERE AND NOWHERE ELSE
---------------------------------
English spelling does not determine pronunciation -- though / through / tough /
thought -- so no rule table can cover words a lexicon has never seen. The
pronunciation lexicons resolve 91.3% of the 20k commonest English words and
95.6% of the evaluation vocabulary; what is left is names, neologisms and
technical terms, which is exactly what a model is for. It runs ONLY after
lookup fails, and its output is tagged `g2p` so its share of any result is
always measurable.

It never sees Thaana. It produces phonemes, and the deterministic rule engine
converts those exactly as it converts a dictionary word, so no decision that a
rule makes is ever handed to a model.

TRAINED ON THE RP INVENTORY, NOT RAW CMUDICT
--------------------------------------------
The targets come from the same Stage 1 stack the engine consumes (en_UK ->
Britfone -> CMUdict with the GenAm->RP transform). Training on raw CMUdict
would emit General American phonemes into tables induced from RP, and we
measured what that costs: 28.0% against 36.4%.

SIZE
----
4 encoder / 2 decoder layers, d_model 256, d_ff 1024, 4 heads, with dedicated
character and phoneme vocabularies rather than a byte vocabulary. About 4M
parameters. At ~1M accuracy drops about 5 points; at ~20M it gains about 2.

    venv/Scripts/python.exe train_g2p.py
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from en2thaana.ipa import (load_britfone, load_cmudict, load_en_uk,  # noqa: E402
                           lookup_backend, phonemes)
from eval.gold import load_gold  # noqa: E402
from eval.splits import split_of  # noqa: E402

OUT = ROOT / "models"
PAD, BOS, EOS = 0, 1, 2
MAX_SRC, MAX_TGT = 26, 30


def build_data():
    """-> [(word, [phoneme, ...])], with gold dev/test words excluded."""
    words = sorted(set(load_en_uk()) | set(load_britfone()) | set(load_cmudict()))
    # Holding out the evaluation words matters: if the model can memorise a
    # dev word's pronunciation, the end-to-end number measures memorisation.
    held = {w for w, _ in load_gold() if split_of(w) != "train"}
    rows = []
    for w in words:
        if not w.isalpha() or not w.isascii() or len(w) > MAX_SRC:
            continue
        if w in held:
            continue
        ipa, _src = lookup_backend(w)
        if not ipa:
            continue
        ph = phonemes(ipa)
        if not ph or len(ph) > MAX_TGT - 2:
            continue
        rows.append((w, ph))
    return rows


class Vocab:
    def __init__(self, symbols):
        self.itos = ["<pad>", "<bos>", "<eos>"] + sorted(symbols)
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)


class G2PData(Dataset):
    def __init__(self, rows, sv, tv):
        self.rows, self.sv, self.tv = rows, sv, tv

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        w, ph = self.rows[i]
        src = [self.sv.stoi[c] for c in w]
        tgt = [BOS] + [self.tv.stoi[p] for p in ph] + [EOS]
        return torch.tensor(src), torch.tensor(tgt)


def collate(batch):
    srcs, tgts = zip(*batch)
    sm = max(len(s) for s in srcs)
    tm = max(len(t) for t in tgts)
    S = torch.full((len(srcs), sm), PAD, dtype=torch.long)
    T = torch.full((len(tgts), tm), PAD, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        S[i, :len(s)] = s
        T[i, :len(t)] = t
    return S, T


class PositionalEncoding(nn.Module):
    def __init__(self, d, maxlen=64):
        super().__init__()
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class G2P(nn.Module):
    def __init__(self, n_src, n_tgt, d=256, ff=1024, heads=4, enc=4, dec=2):
        super().__init__()
        self.d = d
        self.src_emb = nn.Embedding(n_src, d, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(n_tgt, d, padding_idx=PAD)
        self.pos = PositionalEncoding(d)
        self.tf = nn.Transformer(d_model=d, nhead=heads, num_encoder_layers=enc,
                                 num_decoder_layers=dec, dim_feedforward=ff,
                                 dropout=0.1, batch_first=True, norm_first=True)
        self.out = nn.Linear(d, n_tgt)

    def forward(self, src, tgt_in):
        sm = src == PAD
        tm = tgt_in == PAD
        cm = nn.Transformer.generate_square_subsequent_mask(
            tgt_in.size(1), device=tgt_in.device)
        h = self.tf(self.pos(self.src_emb(src) * math.sqrt(self.d)),
                    self.pos(self.tgt_emb(tgt_in) * math.sqrt(self.d)),
                    tgt_mask=cm, src_key_padding_mask=sm,
                    tgt_key_padding_mask=tm, memory_key_padding_mask=sm)
        return self.out(h)

    @torch.no_grad()
    def greedy(self, src, max_len=MAX_TGT):
        self.eval()
        ys = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            nxt = self.forward(src, ys)[:, -1].argmax(-1, keepdim=True)
            nxt[done] = PAD
            ys = torch.cat([ys, nxt], 1)
            done |= nxt.squeeze(1) == EOS
            if done.all():
                break
        return ys


def main():
    OUT.mkdir(exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev}", flush=True)

    rows = build_data()
    sv = Vocab({c for w, _ in rows for c in w})
    tv = Vocab({p for _, ph in rows for p in ph})
    print(f"{len(rows)} pairs | src vocab {len(sv)} | tgt vocab {len(tv)}", flush=True)

    def bucket(w):
        return int(hashlib.sha1(("g2p" + w).encode()).hexdigest(), 16) % 40
    train = [r for r in rows if bucket(r[0]) < 38]
    val = [r for r in rows if bucket(r[0]) == 38]
    test = [r for r in rows if bucket(r[0]) == 39]
    print(f"train {len(train)} | val {len(val)} | test {len(test)}", flush=True)

    model = G2P(len(sv), len(tv)).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"parameters {n_par/1e6:.2f}M", flush=True)

    dl = DataLoader(G2PData(train, sv, tv), batch_size=256, shuffle=True,
                    collate_fn=collate, num_workers=0, drop_last=True)
    vdl = DataLoader(G2PData(val, sv, tv), batch_size=512, collate_fn=collate)

    EPOCHS = 30
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    steps = EPOCHS * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps,
                                                pct_start=0.05)
    lossf = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    best, bad = 0.0, 0
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        tot = 0.0
        for S, T in dl:
            S, T = S.to(dev), T.to(dev)
            logits = model(S, T[:, :-1])
            loss = lossf(logits.reshape(-1, logits.size(-1)), T[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()

        # word-level exact match on val
        ok = n = 0
        for S, T in vdl:
            S = S.to(dev)
            pred = model.greedy(S).cpu()
            for i in range(S.size(0)):
                p = [x for x in pred[i].tolist()[1:] if x not in (PAD, BOS)]
                if EOS in p:
                    p = p[:p.index(EOS)]
                g = [x for x in T[i].tolist()[1:] if x not in (PAD, BOS)]
                if EOS in g:
                    g = g[:g.index(EOS)]
                ok += (p == g)
                n += 1
        acc = ok / n
        print(f"epoch {ep:>2}  loss {tot/len(dl):.4f}  val word-acc {acc:.4f}"
              f"  {time.time()-t0:.0f}s", flush=True)
        if acc > best:
            best, bad = acc, 0
            torch.save({"model": model.state_dict(),
                        "src_itos": sv.itos, "tgt_itos": tv.itos},
                       OUT / "g2p.pt")
            (OUT / "g2p_vocab.json").write_text(
                json.dumps({"src": sv.itos, "tgt": tv.itos}, ensure_ascii=False),
                encoding="utf-8")
        else:
            bad += 1
            if bad >= 3:
                print(f"early stop at epoch {ep}", flush=True)
                break

    print(f"\nbest val word accuracy {best:.4f}", flush=True)
    print(f"saved {OUT/'g2p.pt'}  ({n_par/1e6:.2f}M parameters)", flush=True)


if __name__ == "__main__":
    main()
