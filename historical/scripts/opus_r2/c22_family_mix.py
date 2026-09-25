"""Posterior-weighted mixture over the full hypothesis set (incl. pair-win / partner-win / actor-win families and their
thinned versions), K free.  Per hand: P(in G) = sum_h w_h * firstK_h(hand); components computed from the c21 table (+pair_win).
Builds POST2 on r2j2m (77, for EAP evaluation) and on the B6 base (85 pairs; B6-promoted pairs use the extended c14 table)."""
import numpy as np, pandas as pd, hashlib, json, os
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"
post = json.load(open(f"{OUT}/s95_posterior.json"))["posterior"]
TAG = os.environ.get("TAG", "post2"); BASE = os.environ.get("BASE", "r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv")
# per-hand table: c21 (77 members, has paw/acw) + c14 ext (all 96 pairs) for the rest
E = pd.read_parquet(f"{OUT}/c14_hand_tables_ext.parquet")
c21 = pd.read_parquet(f"{OUT}/c21_hand_tables.parquet")[["slot", "h", "q_HD_paw", "q_dev_paw", "q_HD_acw"]]
E = E.merge(c21, on=["slot", "h"], how="left")
# outcome flags
sp = np.load(f"{D}/s_player.npy", mmap_mode="r"); sw = np.load(f"{D}/s_won.npy", mmap_mode="r")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
hh = E.h.values; sl = E.slot.values; pa = mem[sl // 900, (sl % 900) // 30]; pb = mem[sl // 900, sl % 30]
spH = np.asarray(sp[hh]); sa = np.argmax(spH == pa[:, None], axis=1); sb = np.argmax(spH == pb[:, None], axis=1)
won = np.asarray(sw[hh]); ix = np.arange(len(E)); E["pair_win"] = ((won[ix, sa] > 0) | (won[ix, sb] > 0)).astype(float)
ls = np.asarray(Pt[hh, :, PN.index("last_street")]); E["both_flop"] = ((ls[ix, sa] >= 1) & (ls[ix, sb] >= 1)).astype(float)
# pairs without c21 columns (B6-promoted): approximate paw/acw by pair_win split (0.5 each)
for c_ in ["q_HD_paw", "q_HD_acw"]: E[c_] = E[c_].fillna(E.q_HD * E.pair_win * 0.5)
E["q_dev_paw"] = E.q_dev_paw.fillna(E.q_dev_all * E.pair_win * 0.5)
qB = 1 - (1 - E.q_H2) / np.maximum(1 - E.q_H3, 1e-9); E["q_BOTH"] = (E.q_H3 * qB.clip(0, 1)).clip(0, 1)
E = E.sort_values(["slot", "ts"]).reset_index(drop=True)
def comp(G, h):
    spec = {"HD": ("q_HD", None, 1), "H2": ("q_H2", None, 1), "H3": ("q_H3", None, 1), "HH": ("q_HD", None, 1), "DEVF": ("q_dev_first", None, 1), "DEVA": ("q_dev_all", None, 1),
            "HDxPW": ("q_HD", "pair_win", 1), "HDxFLOP": ("q_HD", "both_flop", 1), "DEVAxPW": ("q_dev_all", "pair_win", 1), "HDx.5": ("q_HD", None, .5), "DEVAx.5": ("q_dev_all", None, .5),
            "DEVFx.5": ("q_dev_first", None, .5), "UNIF": (None, None, 1), "HBOTH": ("q_BOTH", None, 1), "DEVFxPW": ("q_dev_first", "pair_win", 1), "DEVFxPWx.5": ("q_dev_first", "pair_win", .5),
            "DEVAxPWx.5": ("q_dev_all", "pair_win", .5), "HDxPWx.5": ("q_HD", "pair_win", .5), "HDxPWx.25": ("q_HD", "pair_win", .25), "DEVFxPWx.25": ("q_dev_first", "pair_win", .25),
            "HDxPAW": ("q_HD_paw", None, 1), "HDxPAWx.5": ("q_HD_paw", None, .5), "DEVxPAW": ("q_dev_paw", None, 1), "DEVxPAWx.5": ("q_dev_paw", None, .5), "HDxACW": ("q_HD_acw", None, 1)}
    if h in ("M3C", "M1C"):
        flag = ((G.m1 + G.m2) > 0).values if h == "M3C" else (G.m1 > 0).values
        o = np.zeros(len(G)); o[np.flatnonzero(flag)[:5]] = 1.0; return o
    qc, cond, thin = spec[h]
    q = np.full(len(G), 0.1) if qc is None else G[qc].values * thin
    if cond: q = q * G[cond].values
    out = np.zeros(len(q)); dist = np.zeros(6); dist[0] = 1.0
    for i, p in enumerate(q):
        out[i] = p * dist[:5].sum(); nd = dist * (1 - p); nd[1:] += dist[:-1] * p; nd[5] += dist[5] * p; dist = nd
    return out
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hi2id = dict(zip(hidx.hi, hidx.hand_id))
base = pd.read_csv(f"{C}/{BASE}", dtype=str); oth = base[base.predicted_behavior == "other_coordination"].pair_id.tolist()
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet"); pid2sl = c38.set_index("pair_id").slot
groups = {s: G for s, G in E.groupby("slot")}
out = base.set_index("pair_id").copy(); n = 0
for pid in oth:
    s = pid2sl[pid]
    if s not in groups: continue
    G = groups[s]; pm = sum(w * comp(G, h) for h, w in post.items() if w > 1e-4)
    p = G.h.values[np.argsort(-pm, kind="stable")][:5].tolist(); assert len(set(p)) == 5
    out.loc[pid, [f"evidence_hand_{i}" for i in range(1, 6)]] = [hi2id[x] for x in p]; n += 1
o = out.reset_index()[base.columns]; path = f"{C}/r2x_{TAG}_on_{BASE.split('_')[0]}.csv"; o.to_csv(path, index=False)
print(f"{TAG}: rewrote {n} pairs -> {path.split('/')[-1]} sha256 {hashlib.sha256(open(path, 'rb').read()).hexdigest()}")
