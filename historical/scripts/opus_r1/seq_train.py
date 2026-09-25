"""Transformer candidate-hand detector over pair-relative action sequences (GPU)."""
import numpy as np, pandas as pd, torch, torch.nn as nn, time, math, sys, os
from scipy.stats import poisson
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; SEQ = f"{OUT}/seq"
torch.backends.cuda.matmul.allow_tf32 = True
dev = torch.device("cuda")
EPOCHS = int(os.environ.get("EPOCHS", "6")); TAG = os.environ.get("TAG", "t1")
X = np.load(f"{SEQ}/train_X.npy"); Mk = np.load(f"{SEQ}/train_M.npy"); S = np.load(f"{SEQ}/train_S.npy"); meta = pd.read_parquet(f"{SEQ}/train_meta.parquet")
N = len(X); y = meta.ev.values.astype(np.float32); trainable = (meta.ev.values == 1) | ~((meta.pos.values == 1) & (meta.phase.values == 0))
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
class Net(nn.Module):
    def __init__(self, d=128, L=3, H=4):
        super().__init__()
        self.e_st = nn.Embedding(4, 12); self.e_role = nn.Embedding(3, 12); self.e_act = nn.Embedding(4, 12); self.e_lar = nn.Embedding(4, 12)
        self.num = nn.Linear(12, 48); self.proj = nn.Linear(48 + 48, d); self.pos = nn.Embedding(33, d); self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, H, dim_feedforward=2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, L)
        self.static = nn.Sequential(nn.Linear(10, 32), nn.GELU())
        self.head = nn.Sequential(nn.Linear(d + 32, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))
    def forward(self, x, m, s):
        c = x[..., :4].long()
        emb = torch.cat([self.e_st(c[..., 0]), self.e_role(c[..., 1]), self.e_act(c[..., 2]), self.e_lar(c[..., 3])], -1)
        h = self.proj(torch.cat([emb, self.num(x[..., 4:])], -1))
        B = h.shape[0]
        h = torch.cat([self.cls.expand(B, -1, -1), h], 1)
        h = h + self.pos.weight[None, :h.shape[1]]
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=h.device), m == 0], 1)
        z = self.enc(h, src_key_padding_mask=pad)[:, 0]
        return self.head(torch.cat([z, self.static(s)], -1)).squeeze(-1)
def swap(x, s):
    x = x.clone(); s = s.clone()
    r = x[..., 1]; x[..., 1] = torch.where(r == 0, torch.ones_like(r), torch.where(r == 1, torch.zeros_like(r), r))
    l = x[..., 3]; x[..., 3] = torch.where(l == 0, torch.ones_like(l), torch.where(l == 1, torch.zeros_like(l), l))
    s[:, [0, 1, 2, 3, 4, 5, 8, 9]] = s[:, [1, 0, 3, 2, 5, 4, 9, 8]]
    return x, s
Xt = torch.from_numpy(X); Mt = torch.from_numpy(Mk); St = torch.from_numpy(S); yt = torch.from_numpy(y)
fold = meta.fold.values
oof = np.zeros(N, np.float32)
BS = 4096
for f in range(5):
    tr_idx = np.where(trainable & (fold != f))[0]; va_idx = np.where(fold == f)[0]
    pos_idx = tr_idx[y[tr_idx] == 1]; neg_idx = tr_idx[y[tr_idx] == 0]
    net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    steps_per_epoch = math.ceil(len(neg_idx) / BS); total = steps_per_epoch * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=total, pct_start=0.1)
    rng = np.random.RandomState(f)
    for ep in range(EPOCHS):
        net.train(); perm = rng.permutation(neg_idx); tl = 0.0
        for b in range(steps_per_epoch):
            nb = perm[b * BS:(b + 1) * BS]; pb = rng.choice(pos_idx, 256, replace=True)
            idx = np.r_[nb, pb]
            xb = Xt[idx].to(dev, non_blocking=True); mb = Mt[idx].to(dev); sb = St[idx].to(dev); yb = yt[idx].to(dev)
            if rng.rand() < 0.5: xb, sb = swap(xb, sb)
            w = torch.where(yb > 0, torch.tensor(len(neg_idx) / len(pos_idx) * 256 / len(nb) / 8.0, device=dev), torch.tensor(1.0, device=dev))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = net(xb, mb, sb)
            loss = (nn.functional.binary_cross_entropy_with_logits(logit.float(), yb, reduction="none") * w).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); tl += loss.item()
        log(f"fold {f} epoch {ep} loss {tl / steps_per_epoch:.4f}")
    net.eval(); preds = []
    with torch.no_grad():
        for b in range(0, len(va_idx), 16384):
            idx = va_idx[b:b + 16384]
            xb = Xt[idx].to(dev); mb = Mt[idx].to(dev); sb = St[idx].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p1 = torch.sigmoid(net(xb, mb, sb).float()); x2, s2 = swap(xb, sb); p2 = torch.sigmoid(net(x2, mb, s2).float())
            preds.append(((p1 + p2) / 2).cpu().numpy())
    oof[va_idx] = np.concatenate(preds)
    torch.save(net.state_dict(), f"{SEQ}/net_{TAG}_f{f}.pt")
from sklearn.metrics import roc_auc_score, average_precision_score
m = trainable
log("OOF ev-vs-neg AUC", round(roc_auc_score(y[m], oof[m]), 4), "AP", round(average_precision_score(y[m], oof[m]), 4), " GBDT AP", round(average_precision_score(y[m], meta.gbdt.values[m]), 4))
meta["nn"] = oof
P = meta[meta.pos == 1].sort_values(["sl", "ts"]).copy()
def map5(df, col):
    out = []
    for k, g in df.groupby("sl"):
        rel = set(g.h[g.ev == 1]); top = g.sort_values(col, ascending=False).h.values[:5]
        hits = 0; ssum = 0.0
        for i, hh in enumerate(top):
            if hh in rel: hits += 1; ssum += hits / (i + 1)
        out.append((g.fam.iloc[0], ssum / min(5, max(len(rel), 1))))
    r = pd.DataFrame(out, columns=["fam", "ap"]); return round(r.ap.mean(), 4), r.groupby("fam").ap.mean().round(4).to_dict()
negm = (meta.pos.values == 0) | (meta.phase.values == 1)
for col in ["gbdt", "nn"]:
    P[col + "_rank"] = 0.0
q_nn = np.quantile(meta.nn[negm], 0.999); q_g = np.quantile(meta.gbdt[negm], 0.999)
P["blend"] = 0.5 * (P.gbdt / (P.gbdt + (1 - P.gbdt) * 50).clip(lower=1e-9)) + 0.5 * (P.nn / (P.nn + (1 - P.nn) * 50).clip(lower=1e-9))
P["blend_geo"] = np.sqrt(P.gbdt.clip(1e-9) * P.nn.clip(1e-9))
for col in ["gbdt", "nn", "blend", "blend_geo"]:
    P["cum"] = P.groupby("sl")[col].cumsum() - P[col]
    P[col + "_plt5"] = P[col] * poisson.cdf(4, P["cum"])
    print(col, "raw", map5(P, col), " plt5", map5(P, col + "_plt5"), flush=True)
meta.to_parquet(f"{SEQ}/oof_{TAG}.parquet")
