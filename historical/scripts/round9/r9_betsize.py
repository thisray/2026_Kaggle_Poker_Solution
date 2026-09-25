"""Round-9: conditional bet-size anomaly features + combo CV test."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

W = "/home/thisray/projects/260916_Kaggle_Poker_workers/round8-research-20260917"
sys.path.insert(0, f"{W}/code")
from metrics import per_pair

OPUS = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
R8 = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round8_raw_20260917"
a_off = np.load(f"{OPUS}/np/a_off.npy"); a_st = np.load(f"{OPUS}/np/a_st.npy")
a_seat = np.load(f"{OPUS}/np/a_seat.npy"); a_act = np.load(f"{OPUS}/np/a_act.npy")
a_amt = np.load(f"{OPUS}/np/a_amount.npy"); a_tc = np.load(f"{OPUS}/np/a_to_call.npy")
a_pot = np.load(f"{OPUS}/np/a_pot_before.npy"); a_pa = np.load(f"{OPUS}/np/a_players_active.npy")
h_bb = np.load(f"{OPUS}/np/h_bb.npy"); h_phase = np.load(f"{OPUS}/np/h_phase.npy")
nh = len(a_off) - 1

# ---- background histogram over all development hands
rows = []
counts = np.diff(a_off)
hand_of_action = np.repeat(np.arange(nh), counts)
mask = h_phase[hand_of_action] == 0
idx = np.flatnonzero(mask)
st = a_st[idx]; act = a_act[idx]; seat = a_seat[idx]
amt = a_amt[idx].astype(np.float64); tc = a_tc[idx].astype(np.float64)
pot = a_pot[idx].astype(np.float64); pa = a_pa[idx].astype(np.float64)
bb = h_bb[hand_of_action[idx]].astype(np.float64)
aggr = (act == 3) | (act == 4) | ((act == 5) & (amt > tc))
is_allin = (act == 5) & aggr
size = np.where(bb > 0, amt / np.maximum(bb, 1), 0)
pot_ratio = np.where(pot > 0, amt / np.maximum(pot, 1), 0)
spr_bin = np.clip((np.log(np.maximum(pot, 1) / np.maximum(bb, 1))) // 1.2, 0, 3).astype(int)  # rough SPR proxy
po_bin = np.clip((tc / np.maximum(pot, 1) * 3).astype(int), 0, 2)
size_bin = np.digitize(pot_ratio, [0.4, 0.8, 1.2, 2.0])  # 0..4
size_bin = np.where(is_allin, 5, size_bin)
state = (st.astype(int) * 60) + (np.clip(pa, 2, 6).astype(int) - 2) * 12 + spr_bin * 3 + po_bin
key = state[aggr] * 6 + size_bin[aggr]
hist = np.bincount(key, minlength=4 * 60 * 6)
H = hist.reshape(-1, 6) + 0.5  # Laplace smoothing
P = H / H.sum(1, keepdims=True)
print("background actions", int(aggr.sum()), "states", int((H.sum(1) > 6).sum()), flush=True)

# ---- per-row features
FEATS = ["bs_A_sur_max", "bs_A_sur_sum", "bs_A_sur_n", "bs_B_sur_max", "bs_B_sur_sum",
         "bs_B_sur_n", "bs_A_over_sur_max", "bs_B_over_sur_max", "bs_pair_aggr_n"]
pidx = pd.read_parquet(f"{OPUS}/np/player_index.parquet"); p2pi = dict(zip(pidx.player_id, pidx.pi))
hidx = pd.read_parquet(f"{OPUS}/np/hand_index.parquet"); h2hi = dict(zip(hidx.hand_id, hidx.hi))
sp = np.load(f"{OPUS}/np/s_player.npy")
meta = pd.read_csv(f"{R8}/dev_pack/meta.csv")
out = np.zeros((len(meta), len(FEATS)), np.float32)
for n, (hid, plo, phi) in enumerate(zip(meta.hand_id.values, meta.pair_player_lo.values, meta.pair_player_hi.values)):
    hi = h2hi[hid]; seats = sp[hi]
    A = int(np.flatnonzero(seats == p2pi[plo])[0]); B = int(np.flatnonzero(seats == p2pi[phi])[0])
    bbv = float(h_bb[hi]) if h_bb[hi] > 0 else 1.0
    folds = np.zeros(6, bool)
    for k in range(a_off[hi], a_off[hi + 1]):
        act = a_act[k]; i = a_seat[k]
        if act == 0:
            folds[i] = True
            continue
        amt_k = float(a_amt[k]); tc_k = float(a_tc[k]); pot_k = float(a_pot[k]); pa_k = float(a_pa[k])
        aggr_k = (act == 3) or (act == 4) or (act == 5 and amt_k > tc_k)
        if not aggr_k or i not in (A, B):
            continue
        pr = amt_k / max(pot_k, 1.0)
        sb = 5 if (act == 5 and aggr_k) else int(np.digitize(pr, [0.4, 0.8, 1.2, 2.0]))
        sprb = int(np.clip(np.log(max(pot_k, 1.0) / max(bbv, 1.0)) // 1.2, 0, 3))
        pob = int(np.clip(tc_k / max(pot_k, 1.0) * 3, 0, 2))
        stt = int(a_st[k]); pa2 = int(np.clip(pa_k, 2, 6)) - 2
        skey = stt * 60 + pa2 * 12 + sprb * 3 + pob
        p_bin = P[skey, sb]
        sur = -np.log(max(p_bin, 1e-6))
        base = 0.0 if i == A else 3
        out[n, base + 0] = max(out[n, base + 0], sur)
        out[n, base + 1] += sur
        out[n, base + 2] += 1
        out[n, 8] += 1
    # overbet (size_bin==4 or allin) surprisal
    for k in range(a_off[hi], a_off[hi + 1]):
        act = a_act[k]; i = a_seat[k]
        amt_k = float(a_amt[k]); tc_k = float(a_tc[k]); pot_k = float(a_pot[k])
        aggr_k = (act == 3) or (act == 4) or (act == 5 and amt_k > tc_k)
        if not aggr_k or i not in (A, B):
            continue
        pr = amt_k / max(pot_k, 1.0)
        sb = 5 if (act == 5 and aggr_k) else int(np.digitize(pr, [0.4, 0.8, 1.2, 2.0]))
        if sb >= 4:
            sprb = int(np.clip(np.log(max(pot_k, 1.0) / max(bbv, 1.0)) // 1.2, 0, 3))
            pob = int(np.clip(tc_k / max(pot_k, 1.0) * 3, 0, 2))
            skey = int(a_st[k]) * 60 + (int(np.clip(float(a_pa[k]), 2, 6)) - 2) * 12 + sprb * 3 + pob
            sur = -np.log(max(P[skey, sb], 1e-6))
            if i == A:
                out[n, 6] = max(out[n, 6], sur)
            else:
                out[n, 7] = max(out[n, 7], sur)
    if n % 2000 == 0:
        print("betsize", n, "/", len(meta), flush=True)
np.save(f"{R8}/dev_betsize.npy", out)
from sklearn.metrics import roc_auc_score
for j, name in enumerate(FEATS):
    if out[:, j].std() > 1e-9:
        print(f"AUC {name}: {roc_auc_score(meta.ev.values, out[:, j]):.4f}", flush=True)

# ---- combo + betsize CV
SCORES = ['sc_r5','u0','u_rr','u_r5b','lin_contrib','nn_contrib','t1_score','s1_stage1','gen_logit','gen_rank_pct','rank_u_r5b']
r = Path(f"{R8}/dev_pack")
t = np.load(r/'tab.npy', mmap_mode='r', allow_pickle=False)
x = np.nan_to_num(np.c_[t.mean(1), t.max(1)], nan=0., posinf=30., neginf=-30.).clip(-30, 30)
s = meta[SCORES].replace([np.inf,-np.inf], np.nan).fillna(0).to_numpy(float)
ben = np.load(f"{R8}/dev_beneficiary.npy")
X = np.c_[x, s, ben, out]
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
b = meta.u_r5b.to_numpy(); y = meta.ev.to_numpy(int); m = meta.p_fold = meta.m_p.to_numpy(float)
z = np.full(len(meta), np.nan)
for f in sorted(meta.fold.unique()):
    va = meta.fold.to_numpy() == f
    cal = LogisticRegression(C=10, max_iter=1000).fit(b[~va, None], y[~va])
    sl = float(cal.coef_[0, 0]); ic = float(cal.intercept_[0])
    mm = CatBoostClassifier(iterations=400, depth=4, learning_rate=.03, l2_leaf_reg=30,
                            random_seed=71, thread_count=8, verbose=False, allow_writing_files=False)
    mm.fit(Pool(X[~va], y[~va], baseline=sl*b[~va]+ic, weight=1/m[~va]))
    z[va] = b[va] + 0.25 * mm.predict(X[va], prediction_type='RawFormulaVal')/sl
res = {"baseline_E": round(float(per_pair(meta, b).E.mean()), 6),
       "combo_plus_beneficiary_plus_betsize_E": round(float(per_pair(meta, z).E.mean()), 6)}
print(json.dumps(res, indent=2))
json.dump(res, open(f"{R8}/r9_betsize_cv.json", "w"), indent=2)
