"""R4-P1a: select dev pairs for the pair-level event-signal feasibility test and build their frames.
Positives: all labelled dev positives (372). Band pairs: unlabelled, non-hidden dev pairs whose devsub11 OOF rank is in (RLO, RHI]; flag `marker` = contains a player of an
eval top-450 pair (certain dev negative by cross-phase exclusivity). Writes pairs (slot, pa, pb, grp, fam, oof rank) and the R18 gameplay frame for the band pairs."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18"); sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917")
from extract_ci_eval_gb10 import extract
import pairindex as PI
from pathlib import Path
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; RAW = f"{A_}/data/raw"; RLO, RHI = 150, 3000
pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi)); loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
evalp = pd.read_csv(f"{RAW}/evaluation_pairs.csv"); evalp["a"] = evalp.player_1.map(pmap); evalp["b"] = evalp.player_2.map(pmap)
cand = pd.read_csv(f"{OUT}/r2_candidates/r15_grpfuse_rank.csv", usecols=["pair_id", "risk_score"]).merge(evalp[["pair_id", "a", "b"]], on="pair_id"); cand["rk"] = cand.risk_score.rank(ascending=False, method="first")
ep = set(cand[cand.rk <= 450].a) | set(cand[cand.rk <= 450].b)
o = pd.read_parquet(f"{OUT}/m15_o_pos_a_train_oof.parquet"); x = o[o.src == "devsub11"].copy(); x["rank"] = x.oof.rank(ascending=False, method="first")
x["pa"] = x.key // 12000; x["pb"] = x.key % 12000; x["slot"] = x.pool * 900 + loc.local.loc[x.pa].values * 30 + loc.local.loc[x.pb].values
x["marker"] = x.pa.isin(ep) | x.pb.isin(ep)
band = x[(x.label == -1) & (~x.hid.astype(bool)) & (x["rank"] > RLO) & (x["rank"] <= RHI)].copy(); band["grp"] = np.where(band.marker, "neg_marker", "unl")
pos = x[x.label == 1].copy(); pos["grp"] = "pos"
pairs = pd.concat([pos, band])[["slot", "pa", "pb", "grp", "fam", "rank", "oof", "pool", "fold", "n"]].reset_index(drop=True)
print(pairs.grp.value_counts().to_dict(), "| weak positives (rank >", RLO, "):", int(((pairs.grp == "pos") & (pairs["rank"] > RLO)).sum()), flush=True)
pairs.to_parquet(f"{OUT}/r4/p1_pairs.parquet")
H, S, T, SL = PI.all_pair_hands(0); m = np.isin(SL, band.slot.values); K = pd.DataFrame({"slot": SL[m], "h": H[m]}); K["fam"] = "band"; K["pair_id"] = K.slot.astype(str)
full = extract(Path(OUT), K); full.to_parquet(f"{OUT}/r4/p1_band_full.parquet", index=False); print("band frame", full.shape, "pairs", full.slot.nunique())
