"""Are colluders player-disjoint (each colluder has one partner)?  If so, pairs that share a player with a higher-ranked
confident pair are likely false positives -> matching-style demotion for P.  Checks on dev labels, devsub OOF, and eval."""
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as APS
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; RAW = f"{A_}/data/raw"
lab = pd.read_csv(f"{RAW}/development_labels.csv"); pos = lab[lab.label == 1]
pl = pd.concat([pos.player_1, pos.player_2]); vc = pl.value_counts()
print("dev labelled positives:", len(pos), " distinct players", pl.nunique(), " players in >1 positive pair:", int((vc > 1).sum()), " max pairs per player", int(vc.max()))
neg = lab[lab.label == 0]; pn = set(pd.concat([neg.player_1, neg.player_2])); print("positive players that also appear in labelled-negative pairs:", len(set(pl) & pn))
# devsub OOF (v6 base) : rank structure
oof = pd.read_parquet(f"{OUT}/m15_v6base_r2_train_oof.parquet")
pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet"); gi2pid = dict(zip(pidx.pi, pidx.player_id))
for src in ["devsub11", "devsub12"]:
    t = oof[oof.src == src].copy(); t["p_lo"] = (t.key // 12000).astype(int); t["p_hi"] = (t.key % 12000).astype(int)
    t = t.sort_values("oof", ascending=False).reset_index(drop=True); t["rk"] = np.arange(1, len(t) + 1)
    seen = {}; share_prev = []
    for r in t.itertuples():
        s = (r.p_lo in seen) or (r.p_hi in seen)
        share_prev.append(s)
        if r.rk <= 2000:
            seen.setdefault(r.p_lo, r.rk); seen.setdefault(r.p_hi, r.rk)
    t["shares_higher"] = share_prev
    clean = ~t.hid
    top = t[(t.rk <= 1000) & clean]
    g = top.groupby("shares_higher").agg(n=("y", "size"), pos_rate=("y", "mean"))
    print(f"{src}: top-1000 (clean) pairs sharing a player with a higher-ranked pair (within top-2000 seen):\n{g}")
    # demotion: multiply risk of 'shares_higher' pairs by factor f and recompute AP_clean
    base = APS(t.y[clean], t.oof[clean])
    for f in [0.9, 0.5, 0.2, 0.0]:
        s2 = np.where(t.shares_higher, t.oof * f, t.oof)
        print(f"   demote x{f}: AP_clean {APS(t.y[clean], s2[clean]):.4f} (base {base:.4f})")
