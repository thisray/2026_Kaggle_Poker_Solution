"""R3-P18: are the top-ranked `touch_pos` pairs false positives, or unlisted collusion pairs?

Split the hard negatives by how many of their two members are known colluders:
  BOTH  = two known colluders who are not listed as a pair together -> almost certainly an unlisted collusion pair
          (or at minimum an unusable label), because collusion clusters on players;
  ONE   = a known colluder with an otherwise clean partner -> a genuine hard negative.
AP is then recomputed with the BOTH group removed (treated as unknown), which is the honest dev proxy for eval.
Also reports the base rate of BOTH pairs so the concentration at the top can be judged.
"""
import numpy as np, pandas as pd, json
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
def ap(y, s):
    o = np.argsort(-s, kind="mergesort"); r = y[o]; tp = np.cumsum(r); k = np.arange(1, len(r) + 1)
    return float(np.sum(tp / k * r) / max(int(y.sum()), 1))
pidx = pd.read_parquet(f"{O}/np/player_index.parquet"); pm = dict(zip(pidx.player_id, pidx.pi))
lab = pd.read_csv(f"{RAW}/development_labels.csv")
posp = np.array(sorted(set(lab.loc[lab.label == 1, "player_1"].map(pm)) | set(lab.loc[lab.label == 1, "player_2"].map(pm))))
out = {}
for tag in ("m15_o_touch_a", "m15_o_touch_b"):
    for src in ("devsub11", "devsub12"):
        d = pd.read_parquet(f"{O}/{tag}_train_oof.parquet"); M = d[d.src == src].copy()
        k = M.key.values.astype(np.int64); plo = k // 12000; phi = k % 12000
        nc = np.isin(plo, posp).astype(int) + np.isin(phi, posp).astype(int)
        M["nc"] = nc; y = M.y.values.astype(int); hid = M.hid.astype(bool).values
        clean = ~hid | (y == 1)
        hard = (nc > 0) & (M.label.values < 0)
        both = hard & (nc == 2); one = hard & (nc == 1)
        s = M.oof.values
        top450 = M[clean].assign(s=s[clean]).sort_values("s", ascending=False).head(450)
        keep_no_both = clean & ~both
        r = dict(hard_total=int(hard.sum()), hard_both=int(both.sum()), hard_one=int(one.sum()),
                 base_rate_both=round(float(both.sum() / hard.sum()), 4),
                 top450_hard=int(((top450.nc > 0) & (top450.label < 0)).sum()),
                 top450_both=int(((top450.nc == 2) & (top450.label < 0)).sum()),
                 top450_one=int(((top450.nc == 1) & (top450.label < 0)).sum()),
                 ap_restricted=round(ap(y[clean & ~hard], s[clean & ~hard]), 5),
                 ap_full=round(ap(y[clean], s[clean]), 5),
                 ap_drop_both_only=round(ap(y[keep_no_both], s[keep_no_both]), 5))
        # how concentrated is BOTH at the top? expected count if ranked at random among hard negatives
        r["expected_top450_both_if_random"] = round(float(r["top450_hard"] * r["base_rate_both"]), 1)
        out[f"{tag}|{src}"] = r
        print(tag, src, json.dumps(r), flush=True)
json.dump(out, open(f"{O}/r3/t81_hardneg_split.json", "w"), indent=1)
