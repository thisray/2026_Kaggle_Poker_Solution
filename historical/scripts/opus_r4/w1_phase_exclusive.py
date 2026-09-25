"""R4-W1: cross-phase exclusivity of colluding players. Dev colluders (labelled + confidently inferred hidden pairs) vs eval top-ranked pairs:
observed overlap of players vs the chance expectation inside the same pools; list of eval top pairs that touch an inferred dev colluder."""
import numpy as np, pandas as pd
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; RAW = f"{A_}/data/raw"; D = f"{OUT}/np"
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
pidx = pd.read_parquet(f"{D}/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
evalp = pd.read_csv(f"{RAW}/evaluation_pairs.csv"); evalp["a"] = evalp.player_1.map(pmap); evalp["b"] = evalp.player_2.map(pmap)
cand = pd.read_csv(f"{OUT}/r2_candidates/r14_fused_rank.csv", usecols=["pair_id", "risk_score", "predicted_behavior"]).merge(evalp[["pair_id", "a", "b"]], on="pair_id")
cand["rk"] = cand.risk_score.rank(ascending=False, method="first").astype(int)
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
# inferred hidden dev colluders: unlabelled dev pairs with a high OOF in BOTH exposure-matched subsamples (strict) or either (loose)
o = pd.read_parquet(f"{OUT}/m15_o_pos_a_train_oof.parquet"); o = o[o.src.isin(["devsub11", "devsub12"])]
u = o[o.label == -1].pivot_table(index="key", columns="src", values="oof")
for thr in (0.5, 0.3, 0.1):
    strict = u[(u.min(axis=1) >= thr)].index; loose = u[(u.max(axis=1) >= thr)].index
    for nm, keys in (("strict", strict), ("loose", loose)):
        hp = set(np.r_[np.asarray(keys) // 12000, np.asarray(keys) % 12000])
        cand["touch"] = cand.a.isin(hp) | cand.b.isin(hp)
        base = cand.touch.mean()
        out = []
        for lo, hi in [(0, 250), (250, 450), (450, 600), (600, 1000), (1000, 2000), (2000, 5000)]:
            z = cand[(cand.rk > lo) & (cand.rk <= hi)]; out.append(f"({lo},{hi}] {int(z.touch.sum())}/{len(z)} exp {base * len(z):.1f}")
        print(f"thr {thr} {nm}: hidden pairs {len(keys)} players {len(hp)} | eval pairs touching {base:.4f} | " + " | ".join(out))
keys = u[(u.min(axis=1) >= 0.3)].index; hp = set(np.r_[np.asarray(keys) // 12000, np.asarray(keys) % 12000]); cand["touch"] = cand.a.isin(hp) | cand.b.isin(hp)
print(cand[cand.touch & (cand.rk <= 1000)].sort_values("rk")[["pair_id", "rk", "predicted_behavior"]].to_string())
