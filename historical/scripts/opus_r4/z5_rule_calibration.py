"""R4-Z5: how well do purely rule-based two-type proxies (no training) reproduce the REAL evidence lists of the known families on dev? This calibrates how much to trust the same
rules on the fourth family, where no label exists. Rules are orientation-free (either member may be the folder / payer):
  A1 member folds to the partner's aggression and the partner wins;  A2 = A1 and the folder held equity >= 0.5;  B member with equity <= 0.3 put in >= 5bb, did not fold to the partner, partner won
  T1 first five A1 | T2 first five A2 | T3 A2 first then B fill | T5 A1 first then B fill | T6 A2, then A1, then B
Also: per-pair event counts of each pattern for every family vs non-evidence base rates (dev), and for F4 members / controls (eval)."""
import numpy as np, pandas as pd
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; pd.set_option("display.width", 220)
def patterns(d):
    sf = (d.x_s_fold_to_r == 1) & (d.k_pr_won > 0); rf = (d.x_r_fold_to_s == 1) & (d.k_ps_won > 0)
    d["A1"] = sf | rf; d["A2"] = (sf & (d.k_ps_eq_last >= 0.5)) | (rf & (d.k_pr_eq_last >= 0.5))
    d["B"] = ((d.k_ps_eq_last <= 0.3) & (d.x_conS >= 5) & (d.k_pr_won > 0) & (d.x_s_fold_to_r == 0)) | ((d.k_pr_eq_last <= 0.3) & (d.x_conR >= 5) & (d.k_ps_won > 0) & (d.x_r_fold_to_s == 0)); return d
t5 = pd.read_parquet(f"{O}/t5_dev_seq.parquet").rename(columns={"sl": "slot"})[["slot", "h", "fam", "ev", "ts"]]
dev = patterns(t5.merge(pd.read_parquet(f"{O}/r4/x2c_role_dev.parquet").drop(columns=["ts"]), on=["slot", "h"]).merge(pd.read_parquet(f"{O}/r4/x11_kernel_dev.parquet"), on=["slot", "h"]).sort_values(["slot", "ts", "h"]).reset_index(drop=True))
def ap5(pred, tru):
    hits = 0; s = 0.0
    for i, p in enumerate(pred[:5]):
        if p in tru: hits += 1; s += hits / (i + 1)
    return s / min(5, len(tru))
def slate(g, order):
    out = []
    for col in order:
        for h in g.h[g[col]].values:
            if h not in out: out.append(h)
    return out[:5]
RULES = {"T1": ["A1"], "T2": ["A2"], "T3": ["A2", "B"], "T5": ["A1", "B"], "T6": ["A2", "A1", "B"], "T7 (B then A2)": ["B", "A2"]}
for fam, F in dev.groupby("fam"):
    res = {k: [] for k in RULES}
    for sl, g in F.groupby("slot"):
        tru = set(g.h[g.ev].values)
        for k, order in RULES.items(): res[k].append(ap5(slate(g, order), tru))
    e = F[F.ev]; ne = F[~F.ev]
    print(f"{fam:24s} rule AP@5: " + "  ".join(f"{k} {np.mean(v):.3f}" for k, v in res.items()) + f" | evidence share A1 {e.A1.mean():.2f} A2 {e.A2.mean():.2f} B {e.B.mean():.2f} | per pair counts A1 {F.groupby('slot').A1.sum().mean():.1f} A2 {F.groupby('slot').A2.sum().mean():.1f} B {F.groupby('slot').B.sum().mean():.1f} (n {F.groupby('slot').size().mean():.0f}) | non-evidence rates A1 {ne.A1.mean():.4f} A2 {ne.A2.mean():.4f} B {ne.B.mean():.4f}")
z = pd.read_parquet(f"{O}/r4/z1_role.parquet"); zp = pd.read_parquet(f"{O}/r4/z1_pairs.parquet") if False else None
