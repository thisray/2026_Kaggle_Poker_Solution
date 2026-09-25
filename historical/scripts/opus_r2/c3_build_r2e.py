"""Candidate r2e / r2g (RULE=earliest: the earliest 5 hands with first-decision score c>0.05, strongest time prior)
Candidate r2e (evidence variant for the fourth family): per-hand score uses ALL of A's decisions while partner B is in the hand
(first decision + later ones, both directions, max), since later decisions were independently confirmed to carry the
partner-card dependence (docs/30 R2-X1 supp. 9).  Same earliest-first decoder as r2b/r2c_ev.  Base: r2d_p2comb_other_on_r15."""
import numpy as np, pandas as pd, hashlib, json, os
from numba import njit
from scipy.stats import poisson
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; RAW = f"{A_}/data/raw"; DST = f"{OUT}/r2_candidates"
BASE = os.environ.get("BASE", "r15"); sfx = "" if BASE == "r11" else "_on_r15"
base = pd.read_csv(f"{DST}/r2d_p2comb_other{sfx}.csv", dtype=str)
ref_ev = pd.read_csv(f"{DST}/r2d_p2comb_other_ev{sfx}.csv", dtype=str)      # first-decision version, for overlap stats
prom = base[base.predicted_behavior == "other_coordination"].pair_id.tolist()
slotmap = pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv")[["slot", "pair_id"]]; slotmap = slotmap[slotmap.pair_id.isin(prom)]
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_act = np.load(f"{D}/a_act.npy")
Y = np.load(f"{OUT}/dec_Y.npy"); P2 = np.load(f"{OUT}/dec_probs_v2.npy"); ts = np.load(f"{D}/h_ts.npy")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
pfeq = np.asarray(Pt[:, :, PN.index("pf_eq_rand")]).astype(np.float32); MU = float(pfeq.mean())
@njit(cache=True)
def hand_score_first(H, S, T, off, a_seat, a_act, a_st, Y, P2, pfeq, mu, out):
    for r in range(len(H)):
        h = H[r]; ka = -1; kb = -1; best = 0.0
        for k in range(off[h], off[h + 1]):
            if a_st[k] != 0: break
            if a_seat[k] == S[r] and ka < 0: ka = k
            if a_seat[k] == T[r] and kb < 0: kb = k
        for d in range(2):
            k1 = ka if d == 0 else kb; k2 = kb if d == 0 else ka; b = T[r] if d == 0 else S[r]
            if k1 < 0 or (k2 >= 0 and k2 < k1): continue
            v = (((1.0 if Y[k1] == 3 else 0.0) - P2[k1, 3]) - ((1.0 if Y[k1] == 0 else 0.0) - P2[k1, 0])) * (pfeq[h, b] - mu)
            if v > best: best = v
        out[r] = best
@njit(cache=True)
def hand_score(H, S, T, off, a_seat, a_act, Y, P2, pfeq, mu, out):
    for r in range(len(H)):
        h = H[r]; best = 0.0
        for d in range(2):
            a = S[r] if d == 0 else T[r]; b = T[r] if d == 0 else S[r]
            e = pfeq[h, b] - mu; b_active = True; tot = 0.0
            for k in range(off[h], off[h + 1]):
                s = a_seat[k]
                if s == b:
                    if a_act[k] == 0: b_active = False
                    continue
                if s != a: continue
                if not b_active: break
                tot += (((1.0 if Y[k] == 3 else 0.0) - P2[k, 3]) - ((1.0 if Y[k] == 0 else 0.0) - P2[k, 0])) * e
            if tot > best: best = tot
        out[r] = best
H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, slotmap.slot.values); H, S, T, SL = H[m], S[m], T[m], SL[m]
c = np.zeros(len(H))
if os.environ.get("RULE", "all") == "earliest":
    a_st = np.load(f"{D}/a_st.npy"); hand_score_first(H, S, T, off, a_seat, a_act, a_st, Y, P2, pfeq, MU, c)
else:
    hand_score(H, S, T, off, a_seat, a_act, Y, P2, pfeq, MU, c)
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hi2id = dict(zip(hidx.hi, hidx.hand_id))
C = pd.DataFrame({"slot": SL, "h": H, "ts": ts[H], "c": c}).sort_values(["slot", "ts"]).reset_index(drop=True)
C["q"] = np.clip(C.c / 0.3, 0, 1); C["cum"] = C.groupby("slot").q.cumsum() - C.q; C["tpct"] = C.groupby("slot").ts.rank(pct=True)
C["dec"] = C.q * poisson.cdf(3, C.cum) * np.exp(-0.25 * C.tpct); C["hand_id"] = C.h.map(hi2id)
RULE = os.environ.get("RULE", "all")
if RULE == "earliest":
    C["dec"] = np.where(C.c > 0.05, 2.0 - C.tpct, np.where(C.c > 0, 1.0 - C.tpct, -C.tpct))
EV = [f"evidence_hand_{i}" for i in range(1, 6)]
new = base.set_index("pair_id"); refi = ref_ev.set_index("pair_id"); overlap = []
for sl, g in C.groupby("slot"):
    pid = slotmap.set_index("slot").pair_id.loc[sl]
    pick = g.sort_values(["dec", "c"], ascending=False).hand_id.tolist()[:5]
    if len(pick) < 5: pick += [x for x in refi.loc[pid, EV].tolist() if x not in pick and x != "NO_EVIDENCE"][: 5 - len(pick)]
    new.loc[pid, EV] = pick + ["NO_EVIDENCE"] * (5 - len(pick))
    overlap.append(len(set(pick) & set(refi.loc[pid, EV])))
out = new.reset_index()[base.columns]; path = f"{DST}/" + ("r2g_p2comb_other_evearly" if os.environ.get("RULE", "all") == "earliest" else "r2e_p2comb_other_evall") + f"{sfx}.csv"; out.to_csv(path, index=False)
sp = np.load(f"{D}/s_player.npy", mmap_mode="r"); phase = np.load(f"{D}/h_phase.npy", mmap_mode="r"); hid2hi = dict(zip(hidx.hand_id, hidx.hi))
pidx = pd.read_parquet(f"{D}/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi)); evp = pd.read_csv(f"{RAW}/evaluation_pairs.csv").set_index("pair_id")
bad = 0
for r in out[out.pair_id.isin(prom)].itertuples():
    for x in [r.evidence_hand_1, r.evidence_hand_2, r.evidence_hand_3, r.evidence_hand_4, r.evidence_hand_5]:
        if x == "NO_EVIDENCE": continue
        hi = hid2hi[x]; seats = np.asarray(sp[hi])
        if phase[hi] != 1 or pmap[evp.loc[r.pair_id, "player_1"]] not in seats or pmap[evp.loc[r.pair_id, "player_2"]] not in seats: bad += 1
diff = int((out.to_numpy() != ref_ev.to_numpy()).any(1).sum())
chk = dict(rows=len(out), promoted=len(prom), illegal_cells=bad, rows_differing_from_first_decision_version=diff,
           mean_overlap_with_first_decision_picks=float(np.mean(overlap)), sha256=hashlib.sha256(open(path, "rb").read()).hexdigest())
json.dump(chk, open(path.replace(".csv", ".receipt.json"), "w"), indent=1); print(json.dumps(chk, indent=1))
