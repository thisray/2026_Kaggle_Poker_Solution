"""R4-W2: label-free eval-side check of a pair ranking using cross-phase exclusivity. Marker pairs = eval pairs that contain a player of a confidently inferred hidden dev
colluding pair; by exclusivity (0 of the eval top-450 vs 17.8 expected) they are certain negatives. A better ranking places FEWER markers in its top-K.
Reports cumulative marker counts for several K and marker definitions, for each candidate file."""
import numpy as np, pandas as pd, sys
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; RAW = f"{A_}/data/raw"
pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
evalp = pd.read_csv(f"{RAW}/evaluation_pairs.csv"); evalp["a"] = evalp.player_1.map(pmap); evalp["b"] = evalp.player_2.map(pmap)
o = pd.read_parquet(f"{OUT}/m15_o_pos_a_train_oof.parquet"); o = o[o.src.isin(["devsub11", "devsub12"])]; u = o[o.label == -1].pivot_table(index="key", columns="src", values="oof")
KS = (450, 600, 800, 1000, 1500, 2000, 3000)
for nm, keys in (("strict>=0.3", u[u.min(axis=1) >= 0.3].index), ("loose>=0.1", u[u.max(axis=1) >= 0.1].index)):
    hp = set(np.r_[np.asarray(keys) // 12000, np.asarray(keys) % 12000]); evalp["mk"] = evalp.a.isin(hp) | evalp.b.isin(hp); base = evalp.mk.mean()
    print(f"markers {nm}: players {len(hp)}, marker share of eval pairs {base:.4f}")
    for f in sys.argv[1:]:
        c = pd.read_csv(f, usecols=["pair_id", "risk_score", "predicted_behavior"]).merge(evalp[["pair_id", "mk"]], on="pair_id"); c = c[c.predicted_behavior != "other_coordination"]
        c = c.sort_values("risk_score", ascending=False, kind="mergesort").reset_index(drop=True); cum = c.mk.cumsum().values
        print(f"   {f.split('/')[-1]:32s} " + " ".join(f"K{k}:{int(cum[k - 1]):3d}" for k in KS) + f" | expected if all negative beyond 450: " + " ".join(f"{base * (k - 450):.0f}" for k in KS[1:]))
