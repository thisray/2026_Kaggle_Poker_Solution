"""R3-P17: the hard-negative blind spot.

Every pair model excludes `touch_pos & label<0` rows -- pairs where one member is a KNOWN colluder and the pair itself
is not on the official list. Those pairs certainly exist in the 112,540 eval pairs (we cannot know who the colluders
are there), so dev AP is measured on a universe that is missing the hardest negatives, and the models were never
trained to reject them. The TOUCH=1 models kept those rows, so they can measure the effect directly.

Reported per dev subsample: AP on the restricted universe (what every round has quoted) vs AP on the full n>=38
universe, and the composition of the top of the full-universe ranking.
"""
import numpy as np, pandas as pd, json, os
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
def ap(y, s):
    o = np.argsort(-s, kind="mergesort"); r = y[o]; tp = np.cumsum(r); k = np.arange(1, len(r) + 1)
    return float(np.sum(tp / k * r) / max(int(y.sum()), 1))
pidx = pd.read_parquet(f"{O}/np/player_index.parquet"); pm = dict(zip(pidx.player_id, pidx.pi))
lab = pd.read_csv(f"{RAW}/development_labels.csv")
posp = set(lab.loc[lab.label == 1, "player_1"].map(pm)) | set(lab.loc[lab.label == 1, "player_2"].map(pm))
res = {}
for tag in ("m15_o_touch_a", "m15_o_touch_b"):
    f = f"{O}/{tag}_train_oof.parquet"
    if not os.path.exists(f): print("missing", tag); continue
    d = pd.read_parquet(f)
    for src in ("devsub11", "devsub12"):
        M = d[d.src == src].copy()
        k = M.key.values.astype(np.int64); plo = k // 12000; phi = k % 12000
        M["touch"] = np.isin(plo, list(posp)) | np.isin(phi, list(posp))
        y = M.y.values.astype(int); hid = M.hid.astype(bool).values; s = M.oof.values
        restricted = (~M.touch.values) | (M.label.values >= 0)
        clean = ~hid | (y == 1)
        a_rest = ap(y[restricted & clean], s[restricted & clean])
        a_full = ap(y[clean], s[clean])
        top = M[clean].sort_values("oof", ascending=False).head(450)
        extra = int((top.touch & (top.label < 0)).sum())
        # where do the extra hard negatives sit?
        hardneg = M[(M.touch.values) & (M.label.values < 0) & clean]
        rk = pd.Series(s[clean]).rank(ascending=False, pct=True)
        M2 = M[clean].assign(r=pd.Series(s[clean]).rank(ascending=False).values)
        hn = M2[(M2.touch) & (M2.label < 0)]
        res[f"{tag}|{src}"] = dict(rows_full=int(clean.sum()), rows_restricted=int((restricted & clean).sum()),
                                   hard_negatives=int(len(hn)), positives=int(y[clean].sum()),
                                   ap_restricted=round(a_rest, 5), ap_full=round(a_full, 5), drop=round(a_full - a_rest, 5),
                                   hardneg_in_top450=extra, hardneg_in_top1000=int((hn.r <= 1000).sum()),
                                   hardneg_best_rank=int(hn.r.min()) if len(hn) else None,
                                   hardneg_median_rank=int(hn.r.median()) if len(hn) else None)
        print(f"{tag} {src}", json.dumps(res[f"{tag}|{src}"]), flush=True)
json.dump(res, open(f"{O}/r3/t80_touch_blindspot.json", "w"), indent=1)
