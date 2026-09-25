"""Candidate R2-B (NOT submitted): R2-A + evidence for the promoted fourth-family pairs chosen by the per-hand
'informed preflop aggression' contribution c_h = [(aggr - p_aggr) - (fold - p_fold)] * (partner pf_eq - 0.5), decoded with the
LB-validated earliest-first form  q * PoissonCDF(3, sum q_before) * exp(-0.25 tpct)  (same form as u0/r1a)."""
import numpy as np, pandas as pd, hashlib, json, os
from scipy.stats import poisson
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; RAW = f"{A_}/data/raw"
DST = f"{OUT}/r2_candidates"
BASE = os.environ.get("BASE", "r11")
sfx = "" if BASE == "r11" else "_on_r15"
SET = os.environ.get("SET", "r2a")   # r2a (75 pairs) or r2a3 (104 pairs)
base = pd.read_csv(f"{DST}/{ {'r2a': 'r2a_p2_other', 'r2a3': 'r2a3_p2post_other', 'r2c': 'r2c_p2top600_other', 'r2d': 'r2d_p2comb_other'}[SET] }{sfx}.csv", dtype=str)
r11 = pd.read_csv({"r11": f"{A_}/round11_scoped/r11_scoped.csv", "r15": f"{A_}/round15_campaign/r15_tabicl_blend.csv"}[BASE], dtype=str)
prom = base[base.predicted_behavior == "other_coordination"].pair_id.tolist()
slotmap = pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv")[["slot", "pair_id"]]; slotmap = slotmap[slotmap.pair_id.isin(prom)]
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy")
Y = np.load(f"{OUT}/dec_Y.npy"); P2 = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r"); ts = np.load(f"{D}/h_ts.npy")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hi2id = dict(zip(hidx.hi, hidx.hand_id))
H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, slotmap.slot.values); H, S, T, SL = H[m], S[m], T[m], SL[m]
ie = PN.index("pf_eq_rand"); rows = []
for h, s, t, sl in zip(H, S, T, SL):
    ks = {}
    for k in range(off[h], off[h + 1]):
        if a_st[k] != 0: break
        ks.setdefault(int(a_seat[k]), k)
    best = 0.0
    for a, b in [(s, t), (t, s)]:
        if a not in ks or (b in ks and ks[b] < ks[a]): continue
        k = ks[a]; p = np.asarray(P2[k]); eB = float(Pt[h, b, ie])
        best = max(best, ((Y[k] == 3) - p[3] - ((Y[k] == 0) - p[0])) * (eB - 0.5))
    rows.append((sl, h, ts[h], best))
C = pd.DataFrame(rows, columns=["slot", "h", "ts", "c"]).sort_values(["slot", "ts"]).reset_index(drop=True)
C["q"] = np.clip(C.c / 0.3, 0, 1)
C["cum"] = C.groupby("slot").q.cumsum() - C.q
C["tpct"] = C.groupby("slot").ts.rank(pct=True)
C["dec"] = C.q * poisson.cdf(3, C.cum) * np.exp(-0.25 * C.tpct)
C["hand_id"] = C.h.map(hi2id)
EV = [f"evidence_hand_{i}" for i in range(1, 6)]
new = base.set_index("pair_id"); old = r11.set_index("pair_id"); stats = []
for sl, g in C.groupby("slot"):
    pid = slotmap.set_index("slot").pair_id.loc[sl]
    pick = g.sort_values(["dec", "c"], ascending=False).hand_id.tolist()[:5]
    if len(pick) < 5:   # fill from the previous evidence (keeps 5 cells)
        pick += [x for x in old.loc[pid, EV].tolist() if x not in pick and x != "NO_EVIDENCE"][: 5 - len(pick)]
    new.loc[pid, EV] = pick + ["NO_EVIDENCE"] * (5 - len(pick))
    cold = set(old.loc[pid, EV]); stats.append((pid, len(g), int((g.c > 0.1).sum()), len(cold & set(pick)),
                                                float(g.set_index("hand_id").c.reindex(list(cold)).fillna(0).mean()), float(g.set_index("hand_id").c.reindex(pick).fillna(0).mean())))
out = new.reset_index()[base.columns]; stem = {"r2a": "r2b_p2_other_ev", "r2a3": "r2b3_p2post_other_ev", "r2c": "r2c_p2top600_other_ev", "r2d": "r2d_p2comb_other_ev"}[SET]
path = f"{DST}/{stem}{sfx}.csv"; out.to_csv(path, index=False)
st = pd.DataFrame(stats, columns=["pair_id", "hands", "n_c_gt_0.1", "overlap_with_old", "old_mean_c", "new_mean_c"])
print(st.describe().round(3).to_string())
# legality
sp = np.load(f"{D}/s_player.npy", mmap_mode="r"); phase = np.load(f"{D}/h_phase.npy", mmap_mode="r")
hid2hi = dict(zip(hidx.hand_id, hidx.hi)); pidx = pd.read_parquet(f"{D}/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
ev = pd.read_csv(f"{RAW}/evaluation_pairs.csv").set_index("pair_id")
bad = 0; dup = 0
for r in out[out.pair_id.isin(prom)].itertuples():
    cells = [x for x in [r.evidence_hand_1, r.evidence_hand_2, r.evidence_hand_3, r.evidence_hand_4, r.evidence_hand_5] if x != "NO_EVIDENCE"]
    dup += len(cells) != len(set(cells))
    for x in cells:
        hi = hid2hi[x]; seats = np.asarray(sp[hi])
        if phase[hi] != 1 or pmap[ev.loc[r.pair_id, "player_1"]] not in seats or pmap[ev.loc[r.pair_id, "player_2"]] not in seats: bad += 1
chk = dict(rows=len(out), unique=int(out.pair_id.nunique()), promoted=len(prom), illegal_cells=bad, dup_rows=int(dup),
           risk_behavior_identical_to_r2a=bool((out[["risk_score", "predicted_behavior"]].to_numpy() == base[["risk_score", "predicted_behavior"]].to_numpy()).all()),
           evidence_changed_rows=int((out[EV].to_numpy() != base[EV].to_numpy()).any(1).sum()),
           sha256=hashlib.sha256(open(path, "rb").read()).hexdigest())
json.dump(chk, open(f"{DST}/{stem}{sfx}.receipt.json", "w"), indent=1); print(json.dumps(chk, indent=1))
