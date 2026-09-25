"""Local validation of the BF-promotion rule on the exposure-matched dev subsamples (devsub11/12: 2/3 of dev hands, same
masks as the P pipeline).  Pair BF (hand-level tilt mixture, parameters fitted on EVAL members only -> no dev labels used) is
recomputed on each subsample; pairs ranked 600-20000 by the deployed v6 OOF with BF>t are moved to position ~330; paired
AP_raw / AP_clean with vs without promotion.  Dev has no fourth family, so this is a conservative known-family test."""
import numpy as np, pandas as pd, time
from numba import njit
from sklearn.metrics import average_precision_score
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
from bfkern import load_inputs, pair_bf, BETA, PI_
off, a_seat, a_st, Y, Q0, pfeq, eql, MU = load_inputs(); log("inputs")
h_phase = np.load(f"{D}/h_phase.npy")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
T = pd.read_parquet(f"{OUT}/m15_v6ens_base_train_oof.parquet")
H0, S0, T0, SL0 = PI.all_pair_hands(0)
for nm in ["devsub11", "devsub12"]:
    seed = int(nm[-2:]); rng = np.random.RandomState(seed); hm = (h_phase == 0) & (rng.rand(len(h_phase)) < 2 / 3)
    keep = hm[H0]; H, S, Tt, SL = H0[keep], S0[keep], T0[keep], SL0[keep]
    BF = np.zeros(400 * 900); NH = np.zeros(400 * 900); NA = np.zeros(400 * 900)
    pair_bf(H, S, Tt, SL, off, a_seat, a_st, Y, Q0, pfeq, eql, BETA, MU, PI_, BF, NH, NA)
    sl = np.arange(400 * 900); pool = sl // 900; lo = (sl % 900) // 30; hi = sl % 30
    ga = mem[pool, lo]; gb = mem[pool, hi]; key = np.minimum(ga, gb) * 12000 + np.maximum(ga, gb)
    bft = pd.DataFrame({"key": key, "bf": BF, "n_a": NA})[NA > 0]
    t = T[T.src == nm].merge(bft, on="key", how="left"); t["bf"] = t.bf.fillna(-99)
    t = t.sort_values("oof", ascending=False).reset_index(drop=True); t["pos"] = np.arange(len(t))
    lab = np.where(t.y == 1, "pos", np.where(t.hid, "hid", np.where(t.label == 0, "neg", "U")))
    t["lab"] = lab
    log(nm, "pairs", len(t), "BF>3 in top-600:", t.head(600).bf.gt(3).sum(), "labels", t[t.bf > 3].lab.value_counts().to_dict())
    for thr in [3, 4, 5]:
        pm = (t.pos >= 600) & (t.pos < 20000) & (t.bf > thr)
        promo = t[pm].sort_values("bf", ascending=False)
        rest = t[~pm]
        new = pd.concat([rest.iloc[:330], promo, rest.iloc[330:]])
        s_new = np.linspace(1, 0, len(new)); s_old = np.linspace(1, 0, len(t))
        yv_old = t.y.values; yv_new = new.y.values; mc_old = ~t.hid.values; mc_new = ~new.hid.values
        ap_raw0 = average_precision_score(yv_old, s_old); ap_raw1 = average_precision_score(yv_new, s_new)
        ap_c0 = average_precision_score(yv_old[mc_old], s_old[mc_old]); ap_c1 = average_precision_score(yv_new[mc_new], s_new[mc_new])
        print(f"   {nm} BF>{thr}: promoted {len(promo)} (labels {promo.lab.value_counts().to_dict()}, mean old pos {promo.pos.mean():.0f})  AP_raw {ap_raw0:.4f} -> {ap_raw1:.4f} ({ap_raw1 - ap_raw0:+.4f})  AP_clean {ap_c0:.4f} -> {ap_c1:.4f} ({ap_c1 - ap_c0:+.4f})")
