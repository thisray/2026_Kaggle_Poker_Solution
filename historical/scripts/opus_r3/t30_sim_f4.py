"""Self-consistent simulation under the hierarchical substitution model (members' posterior): draw hand planting and decision
activity from the posterior, build the truth under two labeller rules (ACT: first 5 hands with an active decision and the pair
winning; PLANT: first 5 planted hands with the pair winning), score three pickers (r3_main NDw_subh, r3_main_subp, tilt NDw)."""
import numpy as np, pandas as pd
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"; R3 = f"{OUT}/r3"
RHO, SPRE, SPOST = 0.709, 0.508, 0.426
files = {"subh": "r3_main.csv", "subp": "r3_main_subp.csv", "mix": "r3_main_mix.csv", "witness": "r3_main_witness.csv", "tilt": "r3_ctrl_B6tilt.csv"}
P = {k: pd.read_csv(f"{C}/{v}", dtype=str).set_index("pair_id") for k, v in files.items()}
EVC = [f"evidence_hand_{i}" for i in range(1, 6)]
common = [p for p in P["subh"].index[P["subh"].predicted_behavior == "other_coordination"] if all(P[k].loc[p, "predicted_behavior"] == "other_coordination" for k in P)]
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); id2hi = dict(zip(hidx.hand_id, hidx.hi))
e85 = pd.read_parquet(f"{OUT}/s85_eval_bf.parquet")[["slot", "pair_id"]].set_index("pair_id").slot
rows = pd.read_parquet(f"{R3}/t17_rows_eval.parquet"); slots = e85.loc[common].values; rows = rows[rows.slot.isin(slots)]
H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, slots); H, S, T, SL = H[m], S[m], T[m], SL[m]
won = np.load(f"{D}/s_won.npy", mmap_mode="r"); ts = np.load(f"{D}/h_ts.npy"); W = np.asarray(won[H]); ix = np.arange(len(H))
G = pd.DataFrame({"slot": SL, "h": H, "ts": ts[H], "pw": (W[ix, S] > 0) | (W[ix, T] > 0)}).sort_values(["slot", "ts", "h"])
rng = np.random.default_rng(7); NS = 300
res = {(rule, k): [] for rule in ("ACT", "PLANT", "WITNESS") for k in files}
for pid in common:
    sl = e85[pid]; g = G[G.slot == sl]; hs = g.h.values; pw = g.pw.values; pos = {h: i for i, h in enumerate(hs)}
    r = rows[rows.slot == sl]; hi_ = r.h.map(pos).values; s = np.where(r.st.values == 0, SPRE, SPOST); rr = r.r.values
    lf = np.zeros(len(hs)); np.add.at(lf, hi_, np.log((1 - s) + s * rr))
    qpl = RHO * np.exp(lf) / ((1 - RHO) + RHO * np.exp(lf))                 # P(planted | data) per hand
    pd_act = s * rr / ((1 - s) + s * rr)                                      # P(active | planted, data) per decision
    picks = {k: [id2hi[x] for x in P[k].loc[pid, EVC]] for k in files}
    for _ in range(NS):
        planted = rng.random(len(hs)) < qpl
        act_d = rng.random(len(rr)) < pd_act
        act_h = np.zeros(len(hs), bool); np.logical_or.at(act_h, hi_, act_d & planted[hi_])
        chg_d = act_d & planted[hi_] & (rng.random(len(rr)) < np.maximum(1 - 1 / np.maximum(rr, 1e-12), 0))
        wit_h = np.zeros(len(hs), bool); np.logical_or.at(wit_h, hi_, chg_d)
        for rule, flag in (("ACT", act_h & pw), ("PLANT", planted & pw), ("WITNESS", wit_h & pw)):
            truth = set(hs[np.flatnonzero(flag)[:5]])
            if not truth: continue
            for k, pk in picks.items():
                hit = 0; sc = 0.0
                for i, h in enumerate(pk):
                    if h in truth: hit += 1; sc += hit / (i + 1)
                res[(rule, k)].append(sc / min(5, len(truth)))
print(f"pairs {len(common)}, draws {NS} per pair")
for rule in ("ACT", "PLANT", "WITNESS"):
    print(rule, {k: round(float(np.mean(res[(rule, k)])), 4) for k in files})
