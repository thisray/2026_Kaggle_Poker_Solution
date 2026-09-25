"""Step B2 of r5b: transformer inference on candidate sequences (torch env)."""
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(int(os.environ.get("THREADS", "12")))
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
DST = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917"
SEQ = f"{OUT}/seq"
t0 = time.time()


class Net(nn.Module):
    def __init__(self, d=64, L=2, H=4, ns=15):
        super().__init__()
        self.e_st = nn.Embedding(4, 8)
        self.e_role = nn.Embedding(3, 8)
        self.e_act = nn.Embedding(4, 8)
        self.e_lar = nn.Embedding(4, 8)
        self.num = nn.Linear(12, 32)
        self.proj = nn.Linear(64, d)
        self.pos = nn.Embedding(33, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, H, dim_feedforward=2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)
        self.static = nn.Sequential(nn.Linear(ns, 32), nn.GELU())
        self.head = nn.Sequential(nn.Linear(d + 32, 48), nn.GELU(), nn.Dropout(0.2), nn.Linear(48, 1))

    def forward(self, x, m, s):
        c = x[..., :4].long()
        e = torch.cat([self.e_st(c[..., 0]), self.e_role(c[..., 1]), self.e_act(c[..., 2]), self.e_lar(c[..., 3])], -1)
        h = self.proj(torch.cat([e, self.num(x[..., 4:])], -1))
        B = h.shape[0]
        h = torch.cat([self.cls.expand(B, -1, -1), h], 1)
        h = h + self.pos.weight[None, :h.shape[1]]
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool), m == 0], 1)
        z = self.enc(h, src_key_padding_mask=pad)[:, 0]
        return self.head(torch.cat([z, self.static(s)], -1)).squeeze(-1)


lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

meta = pd.read_parquet(f"{DST}/r5cand_meta.parquet").reset_index(drop=True)
X = np.load(f"{DST}/r5cand_X.npy", mmap_mode="r")
M = np.load(f"{DST}/r5cand_M.npy", mmap_mode="r")
S = np.load(f"{DST}/r5cand_S.npy")
N = len(meta)
fam1h = np.stack([(meta.fi.values == k).astype(np.float32) for k in range(3)], 1)
gen = meta.s1.values.astype(np.float32)
gen_rank = meta.gen_rank_all.values.astype(np.float32) if "gen_rank_all" in meta.columns else pd.Series(gen).groupby(meta.slot.values).rank(pct=True).values.astype(np.float32)
Sx = np.concatenate([S, (lg(gen) / 5)[:, None].astype(np.float32), gen_rank[:, None], fam1h], 1).astype(np.float32)
print("static", Sx.shape, flush=True)

preds = np.zeros(N, np.float32)
BS = 8192
for f in range(5):
    net = Net()
    net.load_state_dict(torch.load(f"{SEQ}/seqwithin_f{f}.pt", map_location="cpu"))
    net.eval()
    with torch.no_grad():
        for i in range(0, N, BS):
            xb = torch.from_numpy(np.asarray(X[i:i + BS]))
            mb = torch.from_numpy(np.asarray(M[i:i + BS]))
            sb = torch.from_numpy(Sx[i:i + BS])
            preds[i:i + BS] += torch.sigmoid(net(xb, mb, sb)).numpy()
    print("model", f, "done", round(time.time() - t0, 1), flush=True)
preds /= 5.0
nn_cal = preds / (preds + (1 - preds) * 15.0)
out = meta[["slot", "h"]].copy()
out["nn"] = preds
out["nn_cal"] = nn_cal
out["lg_nn_cal"] = lg(nn_cal)
out.to_parquet(f"{DST}/r5cand_nn.parquet")
print("saved", len(out), "mean nn", float(preds.mean()), "mean cal", float(nn_cal.mean()))
