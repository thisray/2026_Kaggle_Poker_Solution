"""Bayes-optimal evidence under a MIXTURE of labeller hypotheses: per hand, P(hand in G) = sum_k w_k * P_k(hand in G), where
P_k is the first-5 membership probability under hypothesis k (dense-any HD, dense first-two H2, deviation-first, deviation-all,
deterministic M3 core, deterministic M1 core, uniform-early fallback).  Picks the top 5 by the mixture probability.
Usage: WEIGHTS='HD=.25,H2=.15,DEVF=.15,DEVA=.15,M3C=.1,M1C=.05,UNIF=.15' TAG=mixA python c13_mixture_decoder.py"""
import numpy as np, pandas as pd, hashlib, os, pickle
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"
W = dict((k, float(v)) for k, v in (x.split("=") for x in os.environ.get("WEIGHTS", "HD=.25,H2=.15,DEVF=.15,DEVA=.15,M3C=.1,M1C=.05,UNIF=.15").split(",")))
TAG = os.environ.get("TAG", "mixA"); BASE = os.environ.get("BASE", "r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv")
TABLE = os.environ.get("TABLE", "")
NEED_OUTCOMES = any(k in os.environ.get("WEIGHTS", "") for k in ("HDPW", "HDFL"))
if TABLE:
    H = pd.read_parquet(f"{OUT}/c14_hand_tables_{TABLE}.parquet")[["slot", "h", "ts", "q_HD", "q_H2", "q_H3", "q_dev_first", "q_dev_all", "m1", "m2"]].sort_values(["slot", "ts"]).reset_index(drop=True)
else:
    Hq = pd.read_parquet(f"{OUT}/c11_hand_q.parquet")[["slot", "h", "ts", "q_HD", "q_H2", "q_H3", "q_HH"]]
    Hd = pd.read_parquet(f"{OUT}/c12_hand_qdev.parquet")[["slot", "h", "q_dev_first", "q_dev_all"]]
    M = pd.read_parquet(f"{OUT}/s73_f4_mech2.parquet")[["slot", "h", "m1", "m2"]]
    H = Hq.merge(Hd, on=["slot", "h"], how="left").merge(M, on=["slot", "h"], how="left").fillna({"m1": 0, "m2": 0}).sort_values(["slot", "ts"]).reset_index(drop=True)
if NEED_OUTCOMES:
    _sp = np.load(f"{D}/s_player.npy", mmap_mode="r"); _sw = np.load(f"{D}/s_won.npy", mmap_mode="r")
    _Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); _PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
    _loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); _mem = np.zeros((400, 30), np.int64)
    for _pool, _g in _loc.groupby("pool"): _mem[_pool, _g.local.values] = _g.player_gi.values
    _hh = H.h.values; _sl = H.slot.values; _pa = _mem[_sl // 900, (_sl % 900) // 30]; _pb = _mem[_sl // 900, _sl % 30]
    _spH = np.asarray(_sp[_hh]); _sa = np.argmax(_spH == _pa[:, None], axis=1); _sb = np.argmax(_spH == _pb[:, None], axis=1)
    _won = np.asarray(_sw[_hh]); _ix = np.arange(len(H)); H["pair_win"] = ((_won[_ix, _sa] > 0) | (_won[_ix, _sb] > 0)).astype(float)
    _ls = np.asarray(_Pt[_hh, :, _PN.index("last_street")]); H["both_flop"] = ((_ls[_ix, _sa] >= 1) & (_ls[_ix, _sb] >= 1)).astype(float)
def first_k_prob(q, K=5):
    out = np.zeros(len(q)); dist = np.zeros(K + 1); dist[0] = 1.0
    for i, p in enumerate(q):
        out[i] = p * dist[:K].sum(); nd = dist * (1 - p); nd[1:] += dist[:-1] * p; nd[K] += dist[K] * p; dist = nd
    return out
def det_first5(flag):
    idx = np.flatnonzero(flag)[:5]; o = np.zeros(len(flag)); o[idx] = 1.0; return o
comp = {}
for sl, G in H.groupby("slot"):
    c = {"HD": first_k_prob(G.q_HD.values), "H2": first_k_prob(G.q_H2.values), "DEVF": first_k_prob(G.q_dev_first.values),
         "DEVA": first_k_prob(G.q_dev_all.values), "M3C": det_first5(((G.m1 + G.m2) > 0).values), "M1C": det_first5((G.m1 > 0).values),
         "UNIF": first_k_prob(np.full(len(G), 0.1))}
    if NEED_OUTCOMES:
        c["HDPW"] = first_k_prob(G.q_HD.values * G.pair_win.values); c["HDFL"] = first_k_prob(G.q_HD.values * G.both_flop.values)
        c["HDH"] = first_k_prob(0.5 * G.q_HD.values)
    if "BOTH" in W:
        _qB = 1 - (1 - G.q_H2.values) / np.maximum(1 - G.q_H3.values, 1e-9); c["BOTH"] = first_k_prob(np.clip(G.q_H3.values * np.clip(_qB, 0, 1), 0, 1))
    comp[sl] = (G.h.values, c)
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hi2id = dict(zip(hidx.hi, hidx.hand_id))
base = pd.read_csv(f"{C}/{BASE}", dtype=str)
mids = base[base.predicted_behavior == "other_coordination"].pair_id.tolist()
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet"); pid2sl = c38.set_index("pair_id").slot
out = base.set_index("pair_id").copy(); picks = {}; n_changed = 0
for pid in mids:
    sl = pid2sl[pid]
    if sl not in comp: continue                     # pairs without a fourth-family model (e.g. B6-promoted) keep base evidence
    hs, c = comp[sl]; pm = sum(W[k] * c[k] for k in W)
    p = hs[np.argsort(-pm, kind="stable")][:5].tolist(); assert len(set(p)) == 5
    out.loc[pid, [f"evidence_hand_{i}" for i in range(1, 6)]] = [hi2id[x] for x in p]; picks[sl] = p; n_changed += 1
o = out.reset_index()[base.columns]; path = f"{C}/r2x_{TAG}_on_{BASE.split('_')[0]}.csv"; o.to_csv(path, index=False)
print("weights", W, "| pairs changed", n_changed, "| file", path.split("/")[-1], "sha256", hashlib.sha256(open(path, "rb").read()).hexdigest())
# expected AP of this pick set and of the reference variants under each (single) hypothesis
ref = pickle.load(open(f"{OUT}/c12_picks.pkl", "rb")); ref["MIX"] = picks
H = H[H.slot.isin(list(ref['c-first'].keys()))]
rng = np.random.default_rng(5); rows = {}
qcols = {"HD": "q_HD", "H2": "q_H2", "DEVF": "q_dev_first", "DEVA": "q_dev_all"}
for hyp in ["HD", "H2", "DEVF", "DEVA", "M3C", "M1C"]:
    acc = {k: [] for k in ["c-first", "ND", "NDdev", "M3", "M1", "MIX"]}
    for sl, G in H.groupby("slot"):
        hs = G.h.values
        sims = 1 if hyp in ("M3C", "M1C") else 150
        for s in range(sims):
            if hyp == "M3C": ev = hs[((G.m1 + G.m2) > 0).values][:5]
            elif hyp == "M1C": ev = hs[(G.m1 > 0).values][:5]
            else: ev = hs[rng.random(len(hs)) < G[qcols[hyp]].values][:5]
            if len(ev) == 0: continue
            evs = set(ev); den = min(5, len(ev))
            for k in acc:
                hits = 0; ap = 0.0
                for i, x in enumerate(ref[k][sl][:5]):
                    if x in evs: hits += 1; ap += hits / (i + 1)
                acc[k].append(ap / den)
    rows[hyp] = {k: round(float(np.mean(v)), 3) for k, v in acc.items()}
T = pd.DataFrame(rows).T; T.loc["prior-weighted"] = [sum(W.get(h, 0) * T.loc[h, k] for h in T.index if h in W) / sum(W.get(h, 0) for h in T.index if h in W) for k in T.columns]
print(T.round(3).to_string())
