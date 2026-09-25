"""CPU transformer within-pair candidate ranker over action sequences (positive-pair windows only); blended with the GBDT within ranker."""
import numpy as np, pandas as pd, torch, torch.nn as nn, time, math, os
from scipy.stats import poisson
torch.set_num_threads(int(os.environ.get("THREADS", "8")))
OUT = os.environ["POKER_WORK_DIR"]; SEQ = os.environ.get("POKER_SEQ_DIR", f"{OUT}/seq")
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
X = np.load(f"{SEQ}/train_X.npy", mmap_mode="r"); M = np.load(f"{SEQ}/train_M.npy", mmap_mode="r"); S = np.load(f"{SEQ}/train_S.npy", mmap_mode="r")
meta = pd.read_parquet(f"{SEQ}/train_meta.parquet")
idx = np.where((meta.pos.values == 1) & (meta.phase.values == 0))[0]
P = meta.iloc[idx].reset_index(drop=True); P["ev"] = P.ev.astype(bool)
gb_source = os.environ.get("POKER_SEQ_GB_OOF", "m25e1_handfeat2_m19w10_oof.parquet")
gb = pd.read_parquet(f"{OUT}/{gb_source}").set_index(["sl", "h"])
gen = pd.read_parquet(f"{OUT}/m19w10_handscores.parquet").set_index(["sl", "h"]).s
mi = pd.MultiIndex.from_arrays([P.sl.values, P.h.values])
P["gbw"] = gb.sc_fam.reindex(mi).values; P["gen"] = gen.reindex(mi).values
P["gen_rank"] = P.groupby("sl").gen.rank(pct=True)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
fam1h = np.stack([(P.fam.values == f).astype(np.float32) for f in ["directed_transfer", "soft_play", "coordinated_isolation"]], 1)
Sx = np.concatenate([np.asarray(S[idx]), lg(P.gen.values)[:, None].astype(np.float32) / 5, P.gen_rank.values[:, None].astype(np.float32), fam1h], 1).astype(np.float32)
Xt = torch.from_numpy(np.asarray(X[idx])); Mt = torch.from_numpy(np.asarray(M[idx])); St = torch.from_numpy(Sx); yt = torch.from_numpy(P.ev.values.astype(np.float32))
last = P[P.ev].groupby("sl").ts.max(); win = (P.ts.values <= P.sl.map(last).values)
fold = P.fold.values
class Net(nn.Module):
    def __init__(self, d=64, L=2, H=4, ns=Sx.shape[1]):
        super().__init__()
        self.e_st = nn.Embedding(4, 8); self.e_role = nn.Embedding(3, 8); self.e_act = nn.Embedding(4, 8); self.e_lar = nn.Embedding(4, 8)
        self.num = nn.Linear(12, 32); self.proj = nn.Linear(64, d); self.pos = nn.Embedding(33, d); self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, H, dim_feedforward=2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)
        self.static = nn.Sequential(nn.Linear(ns, 32), nn.GELU())
        self.head = nn.Sequential(nn.Linear(d + 32, 48), nn.GELU(), nn.Dropout(0.2), nn.Linear(48, 1))
    def forward(self, x, m, s):
        c = x[..., :4].long()
        e = torch.cat([self.e_st(c[..., 0]), self.e_role(c[..., 1]), self.e_act(c[..., 2]), self.e_lar(c[..., 3])], -1)
        h = self.proj(torch.cat([e, self.num(x[..., 4:])], -1)); B = h.shape[0]
        h = torch.cat([self.cls.expand(B, -1, -1), h], 1); h = h + self.pos.weight[None, :h.shape[1]]
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool), m == 0], 1)
        z = self.enc(h, src_key_padding_mask=pad)[:, 0]
        return self.head(torch.cat([z, self.static(s)], -1)).squeeze(-1)
def swap(x, s):
    x = x.clone(); s = s.clone()
    r = x[..., 1]; x[..., 1] = torch.where(r == 0, torch.ones_like(r), torch.where(r == 1, torch.zeros_like(r), r))
    l = x[..., 3]; x[..., 3] = torch.where(l == 0, torch.ones_like(l), torch.where(l == 1, torch.zeros_like(l), l))
    s[:, [0, 1, 2, 3, 4, 5, 8, 9]] = s[:, [1, 0, 3, 2, 5, 4, 9, 8]]
    return x, s
EPOCHS = int(os.environ.get("EPOCHS", "12")); BS = 256
oof = np.zeros(len(P), np.float32)
for f in range(5):
    torch.manual_seed(f)
    tr = np.where(win & (fold != f))[0]; va = np.where(fold == f)[0]
    net = Net(); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    steps = math.ceil(len(tr) / BS) * EPOCHS; sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.15)
    pos_w = torch.tensor((1 - yt[tr].mean()) / yt[tr].mean())
    rng = np.random.RandomState(f)
    for ep in range(EPOCHS):
        net.train(); perm = rng.permutation(tr); tl = 0.0
        for b in range(0, len(perm), BS):
            ib = perm[b:b + BS]; xb, mb, sb, yb = Xt[ib], Mt[ib], St[ib], yt[ib]
            if rng.rand() < 0.5: xb, sb = swap(xb, sb)
            loss = nn.functional.binary_cross_entropy_with_logits(net(xb, mb, sb), yb, pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); tl += loss.item()
    net.eval()
    with torch.no_grad():
        p1 = torch.sigmoid(net(Xt[va], Mt[va], St[va])); x2, s2 = swap(Xt[va], St[va]); p2 = torch.sigmoid(net(x2, Mt[va], s2))
    oof[va] = ((p1 + p2) / 2).numpy()
    torch.save(net.state_dict(), f"{SEQ}/seqwithin_f{f}.pt")
    log("fold", f, "done, last loss", round(tl / math.ceil(len(tr) / BS), 4))
P["nn"] = oof
def map5(col):
    aps = []
    for k, g in P.groupby("sl"):
        rel = set(g.h[g.ev]); top = g.sort_values(col, ascending=False).h.values[:5]; hits = 0; ssum = 0.0
        for i, hh in enumerate(top):
            if hh in rel: hits += 1; ssum += hits / (i + 1)
        aps.append(ssum / min(5, max(len(rel), 1)))
    return round(float(np.mean(aps)), 4)
P = P.sort_values(["sl", "ts"])
# the nn is trained with pos_weight, so its outputs are inflated; recalibrate with a monotone rank->window-rate map before the Poisson prior
for col in ["gbw", "nn"]:
    P[col + "_blend"] = P[col]
P["nn_cal"] = P.nn / (P.nn + (1 - P.nn) * float(os.environ.get("NNCAL", "15")))
P["blend"] = 1 / (1 + np.exp(-(0.5 * lg(P.gbw.values) + 0.5 * lg(P.nn_cal.values))))
for col in ["gbw", "nn_cal", "blend"]:
    P["cum"] = P.groupby("sl")[col].cumsum() - P[col]; P["x"] = P[col] * poisson.cdf(4, P.cum)
    print(col, "raw", map5(col), "plt5", map5("x"), flush=True)
P[["sl", "h", "ev", "fam", "ts", "gbw", "nn", "nn_cal"]].to_parquet(f"{OUT}/seqwithin_oof.parquet")
