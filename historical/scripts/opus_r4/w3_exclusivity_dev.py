"""R4-W3: dev-side validation of exclusivity-based demotion (the mirror image of what would be applied to eval).
Markers on dev = dev pairs containing a player of a confident EVAL colluding pair (eval top-K of the current best ranking, fourth-family members included).
By cross-phase exclusivity they are certain dev negatives. Measure (a) how many labelled dev positives are markers (should be ~0), (b) AP_clean before / after sending markers to the bottom."""
import numpy as np, pandas as pd, sys
from sklearn.metrics import average_precision_score
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; RAW = f"{A_}/data/raw"
pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
evalp = pd.read_csv(f"{RAW}/evaluation_pairs.csv"); evalp["a"] = evalp.player_1.map(pmap); evalp["b"] = evalp.player_2.map(pmap)
cand = pd.read_csv(f"{OUT}/r2_candidates/r15_grpfuse_rank.csv", usecols=["pair_id", "risk_score"]).merge(evalp[["pair_id", "a", "b"]], on="pair_id"); cand["rk"] = cand.risk_score.rank(ascending=False, method="first")
o = pd.read_parquet(f"{OUT}/{sys.argv[1] if len(sys.argv) > 1 else 'm15_o_pos_a'}_train_oof.parquet"); o["p_lo"] = o.key // 12000; o["p_hi"] = o.key % 12000
for K in (300, 400, 450, 500):
    ep = set(cand[cand.rk <= K].a) | set(cand[cand.rk <= K].b)
    for src in ("devsub11", "devsub12"):
        x = o[o.src == src].copy(); x["mk"] = x.p_lo.isin(ep) | x.p_hi.isin(ep); clean = (~x.hid.astype(bool)) | (x.y == 1); xc = x[clean]
        ap0 = average_precision_score(xc.y, xc.oof); ap1 = average_precision_score(xc.y, np.where(xc.mk, -1.0, xc.oof)); x["rank"] = x.oof.rank(ascending=False, method="first")
        print(f"K={K} {src}: eval-colluder players {len(ep)} | marker share {x.mk.mean():.4f} | labelled positives that are markers {int((x.mk & (x.y == 1)).sum())}/{int((x.y == 1).sum())} | hidden that are markers {int((x.mk & x.hid.astype(bool)).sum())}/{int(x.hid.sum())}"
              f" | markers in top 600 / 1000: {int((x.mk & (x['rank'] <= 600)).sum())}/{int((x.mk & (x['rank'] <= 1000)).sum())} | AP_clean {ap0:.5f} -> {ap1:.5f} ({ap1 - ap0:+.5f})")
