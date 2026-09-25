"""R4-U6: does policy_v1 memorisation of member decisions causally reduce the R15 hit rate? in1 is a random per-decision mask -> natural experiment.
For every dev candidate hand: member decisions, v1 surprisal (as used), v2/v1 out-of-sample surprisal, memorised flags. Control for the number of decisions."""
import numpy as np, pandas as pd
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
pd.set_option("display.width", 250)
hi = pd.read_parquet(f"{D}/hand_index.parquet"); hmap = dict(zip(hi.hand_id, hi.hi))
t = pd.read_parquet(f"{OUT}/r3/t45_known_e_rerank.parquet"); t["h"] = t.hand_id.map(hmap); t["rk"] = t.groupby("slot").b.rank(ascending=False, method="first")
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy", mmap_mode="r"); sp = np.load(f"{D}/s_player.npy", mmap_mode="r")
Y = np.load(f"{OUT}/dec_Y.npy", mmap_mode="r"); P1 = np.load(f"{OUT}/dec_probs_v1.npy", mmap_mode="r"); P2 = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r")
N = int(off[-1]); in1 = np.zeros(N, bool); in1[np.random.RandomState(0).choice(N, 4_000_000, replace=False)] = True
in2 = np.zeros(N, bool); in2[np.random.RandomState(1).choice(N, 6_000_000, replace=False)] = True
print("N decisions", N, "v1 frac", in1.mean(), "v2 frac", in2.mean(), "Y shape", Y.shape, "P1 shape", P1.shape)
rows = []
for h, pa, pb in zip(t.h.values, t.pa.values, t.pb.values):
    seats = np.asarray(sp[h]); sa = int(np.flatnonzero(seats == pa)[0]); sb = int(np.flatnonzero(seats == pb)[0])
    ks = np.arange(off[h], off[h + 1]); seat = np.asarray(a_seat[ks]); km = ks[(seat == sa) | (seat == sb)]
    y = np.asarray(Y[km]); p1 = np.asarray(P1[km])[np.arange(len(km)), y]; p2 = np.asarray(P2[km])[np.arange(len(km)), y]
    s1 = -np.log(np.clip(p1, 1e-6, 1)); s2 = -np.log(np.clip(p2, 1e-6, 1)); m1 = in1[km]; m2 = in2[km]
    # "key" decision = the most surprising one according to the v2 model where v2 is out-of-sample, else v1 out-of-sample
    clean = np.where(~m2, s2, np.where(~m1, s1, np.nan))
    j = int(np.nanargmax(clean)) if np.isfinite(clean).any() else -1
    rows.append((len(km), m1.mean() if len(km) else 0, s1.max() if len(km) else 0, np.nanmax(clean) if j >= 0 else np.nan, bool(m1[j]) if j >= 0 else False,
                 s1[j] if j >= 0 else np.nan, clean[j] if j >= 0 else np.nan))
for j, c in enumerate(["nd", "memo1", "s1max", "cmax", "key_memo1", "key_s1", "key_clean"]): t[c] = [r[j] for r in rows]
e = t[t.ev == 1].copy(); e["hit"] = e.rk <= 5
print("evidence hands", len(e), "hit", e.hit.mean())
print("hit by key decision memorised by v1:"); print(e.groupby(["behavior_family", "key_memo1"]).hit.agg(["mean", "size"]).round(3))
print("key decision surprisal: v1-as-used vs clean, by memorised flag"); print(e.groupby("key_memo1")[["key_s1", "key_clean"]].mean().round(3))
# non-evidence candidates: false-pick rate by flag
n = t[t.ev == 0].copy(); n["fp"] = n.rk <= 5
print("non-evidence false-pick rate by key_memo1"); print(n.groupby(["behavior_family", "key_memo1"]).fp.agg(["mean", "size"]).round(3))
e["ndb"] = pd.cut(e.nd, [0, 2, 4, 6, 8, 100]); print(e.groupby(["ndb", "key_memo1"]).hit.agg(["mean", "size"]).round(3).unstack())
t.to_parquet(f"{OUT}/r4/u6_memo.parquet")
