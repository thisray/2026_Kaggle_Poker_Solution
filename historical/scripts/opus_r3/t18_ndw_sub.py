"""Fourth-family evidence with the exact substitution mechanism: per hand q = 1 - prod_d (1 - post_d) (post_d from t17), NDw_sub =
first-5 DP over q * pair_win (tie-break by q).  Applied to every other_coordination pair of BASE; reports overlap with the tilt NDw.
Usage: BASE=r2f_NDw_all_on_r2j2mB6.csv python t18_ndw_sub.py"""
import numpy as np, pandas as pd, hashlib, os
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"; R3 = f"{OUT}/r3"
BASE = os.environ.get("BASE", "r2f_NDw_all_on_r2j2mB6.csv"); TAG = os.environ.get("TAG", "r3a_NDwsub")
base = pd.read_csv(f"{C}/{BASE}", dtype=str)
e85 = pd.read_parquet(f"{OUT}/s85_eval_bf.parquet")[["slot", "pair_id"]]; pid2sl = e85.set_index("pair_id").slot
oth = base[base.predicted_behavior == "other_coordination"].pair_id.tolist(); slots = pid2sl.loc[oth].values
rows = pd.read_parquet(f"{R3}/t17_rows_eval.parquet"); rows = rows[rows.slot.isin(slots)]
hq = rows.groupby(["slot", "h"]).post.apply(lambda p: 1 - np.prod(1 - p.values)).rename("q").reset_index()
H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, slots); H, S, T, SL = H[m], S[m], T[m], SL[m]
won = np.load(f"{D}/s_won.npy", mmap_mode="r"); ts = np.load(f"{D}/h_ts.npy")
W = np.asarray(won[H]); ix = np.arange(len(H)); pw = (W[ix, S] > 0) | (W[ix, T] > 0)
G = pd.DataFrame({"slot": SL, "h": H, "ts": ts[H], "pair_win": pw}).merge(hq, on=["slot", "h"], how="left").fillna({"q": 0.0})
G = G.sort_values(["slot", "ts", "h"]).reset_index(drop=True)
def first_k_prob(q, K=5):
    out = np.zeros(len(q)); dist = np.zeros(K + 1); dist[0] = 1.0
    for i, p in enumerate(q):
        out[i] = p * dist[:K].sum(); nd = dist * (1 - p); nd[1:] += dist[:-1] * p; nd[K] += dist[K] * p; dist = nd
    return out
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hi2id = dict(zip(hidx.hi, hidx.hand_id)); id2hi = dict(zip(hidx.hand_id, hidx.hi))
out = base.set_index("pair_id").copy(); ov = []; qstat = []
for pid in oth:
    g = G[G.slot == pid2sl[pid]]
    sc = first_k_prob(g.q.values * g.pair_win.values) + 1e-6 * first_k_prob(g.q.values)
    p = g.h.values[np.argsort(-sc, kind="stable")][:5].tolist(); assert len(set(p)) == 5
    old = [id2hi[x] for x in base.set_index("pair_id").loc[pid, [f"evidence_hand_{i}" for i in range(1, 6)]]]
    ov.append(len(set(p) & set(old))); qstat.append(np.mean(np.abs(g.q.values - 0.5)))
    out.loc[pid, [f"evidence_hand_{i}" for i in range(1, 6)]] = [hi2id[x] for x in p]
o = out.reset_index()[base.columns]; path = f"{C}/{TAG}_on_{BASE.replace('.csv', '')}.csv"; o.to_csv(path, index=False)
b2 = base.set_index("pair_id"); chk = o.set_index("pair_id")
assert (chk.risk_score == b2.risk_score).all() and (chk.predicted_behavior == b2.predicted_behavior).all()
print(f"{TAG}: other pairs {len(oth)}; mean overlap with BASE evidence {np.mean(ov):.2f}/5 (dist {np.bincount(ov, minlength=6).tolist()}); mean |q-0.5| {np.mean(qstat):.3f}")
print(path.split("/")[-1], "sha256", hashlib.sha256(open(path, "rb").read()).hexdigest())
