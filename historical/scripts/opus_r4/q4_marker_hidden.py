"""R4-Q4: marker-aware PU cleaning list for the pair model. Unlabelled dev pairs above the usual threshold (ref OOF > 0.3) stay 'hidden'; pairs with a mid score (ref OOF > THR)
are added UNLESS they contain a player of an eval top-450 pair (then they are certain negatives by cross-phase exclusivity and remain clean negatives).
Marker statistics say the (0.05, 0.3] band is mostly hidden positives (0-1 markers seen vs 2-3 expected) that are currently trained as negatives."""
import pandas as pd, numpy as np, sys
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; O = f"{A_}/opus_r1_20260917"; RAW = f"{A_}/data/raw"
pidx = pd.read_parquet(f"{O}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
evalp = pd.read_csv(f"{RAW}/evaluation_pairs.csv"); evalp["a"] = evalp.player_1.map(pmap); evalp["b"] = evalp.player_2.map(pmap)
cand = pd.read_csv(f"{O}/r2_candidates/r15_grpfuse_rank.csv", usecols=["pair_id", "risk_score"]).merge(evalp[["pair_id", "a", "b"]], on="pair_id"); cand["rk"] = cand.risk_score.rank(ascending=False, method="first")
ep = set(cand[cand.rk <= 450].a) | set(cand[cand.rk <= 450].b)
ref = pd.read_parquet(f"{O}/m5_both_train_oof.parquet"); ref = ref[ref.src.isin(["devsub11", "devsub12"]) & (ref.label == -1)].copy(); ref["mk"] = (ref.key // 12000).isin(ep) | (ref.key % 12000).isin(ep)
for thr, nm in ((0.05, "005"), (0.02, "002")):
    h = ref[(ref.oof > 0.3) | ((ref.oof > thr) & ~ref.mk)][["src", "key"]]; h.to_parquet(f"{O}/r4/hid_thr{nm}_nomarker.parquet")
    print(thr, h.groupby("src").size().to_dict(), "| baseline > 0.3:", ref[ref.oof > 0.3].groupby("src").size().to_dict())
