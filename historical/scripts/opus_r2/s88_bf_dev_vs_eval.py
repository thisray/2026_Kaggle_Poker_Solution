"""Mid-rank BF>t pairs: dev subsample (no fourth family) vs eval.  Compare counts by threshold and their first-decision p2
(devsub p2 from s23_infoshare_devsubXX; eval p2 from s23_infoshare_eval)."""
import numpy as np, pandas as pd, time
import pairindex as PI
from bfkern import load_inputs, pair_bf, BETA, PI_
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
off, a_seat, a_st, Y, Q0, pfeq, eql, MU = load_inputs()
h_phase = np.load(f"{D}/h_phase.npy")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
T = pd.read_parquet(f"{OUT}/m15_v6ens_base_train_oof.parquet")
H0, S0, T0, SL0 = PI.all_pair_hands(0)
rows = []
for nm in ["devsub11", "devsub12"]:
    seed = int(nm[-2:]); rng = np.random.RandomState(seed); hm = (h_phase == 0) & (rng.rand(len(h_phase)) < 2 / 3)
    keep = hm[H0]; BF = np.zeros(400 * 900); NH = np.zeros(400 * 900); NA = np.zeros(400 * 900)
    pair_bf(H0[keep], S0[keep], T0[keep], SL0[keep], off, a_seat, a_st, Y, Q0, pfeq, eql, BETA, MU, PI_, BF, NH, NA)
    sl = np.arange(400 * 900); ga = mem[sl // 900, (sl % 900) // 30]; gb = mem[sl // 900, sl % 30]
    bft = pd.DataFrame({"slot": sl, "key": np.minimum(ga, gb) * 12000 + np.maximum(ga, gb), "bf": BF, "n_a": NA})[NA > 0]
    p2 = pd.read_parquet(f"{OUT}/s23_infoshare_{nm}.parquet"); p2["p2"] = np.minimum(p2.za0 - p2.zf0, p2.za1 - p2.zf1)
    t = T[T.src == nm].merge(bft, on="key", how="inner").merge(p2[["slot", "p2"]], on="slot", how="left")
    t = t.sort_values("oof", ascending=False).reset_index(drop=True); t["pos"] = np.arange(len(t)); t["set"] = nm; rows.append(t)
dv = pd.concat(rows)
e = pd.read_parquet(f"{OUT}/s85_eval_bf.parquet"); p2e = pd.read_parquet(f"{OUT}/s23_infoshare_eval.parquet"); p2e["p2"] = np.minimum(p2e.za0 - p2e.zf0, p2e.za1 - p2e.zf1)
e = e.merge(p2e[["slot", "p2"]], on="slot", how="left"); e["pos"] = e.rk_r2j2m - 1
e = e[~e.member]
for t in [3, 4, 5, 6, 8]:
    dmid = dv[(dv.pos >= 600) & (dv.pos < 20000) & (dv.bf > t)]; emid = e[(e.pos >= 600) & (e.pos < 20000) & (e.bf > t)]
    print(f"BF>{t}: devsub mean count {len(dmid) / 2:.1f} (p2 median {dmid.p2.median():.2f}) | eval {len(emid)} (p2 median {emid.p2.median():.2f})"
          f" | devsub top-600 non-pos count {len(dv[(dv.pos < 600) & (dv.bf > t) & (dv.y == 0)]) / 2:.1f}")
dmid = dv[(dv.pos >= 600) & (dv.pos < 20000) & (dv.bf > 3)]
print("devsub mid BF>3: p2 quantiles", np.round(dmid.p2.quantile([.1, .5, .9]).values, 2), " n_a median", dmid.n_a.median(), " | eval mid BF>3 n_a median", e[(e.pos >= 600) & (e.pos < 20000) & (e.bf > 3)].n_a.median())
print("devsub: BF of labelled positives (all ranks):", dv[dv.y == 1].bf.describe()[["mean", "50%"]].round(2).to_dict(), " share >3:", round((dv[dv.y == 1].bf > 3).mean(), 3))
