"""Round-10 E1: evidence top-5 selection experiments on dev_pack.

Variants (all 5-fold OOF, seed 71, identical recipe to round-8 combo):
  A moments+scores (replication)
  B A + narrow extras
  C B + board/SPR features
  D B + set-consistency features
  E B + C + D
  F XGBRanker rank:pairwise on E features
  G blend z_E + w * z_F
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from scipy.special import expit

sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/round8-research-20260917/code")
from metrics import per_pair

R8 = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round8_raw_20260917"
OP = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
OUT = Path(f"{R8}/r10_e1")
OUT.mkdir(parents=True, exist_ok=True)

SCORES = ['sc_r5', 'u0', 'u_rr', 'u_r5b', 'lin_contrib', 'nn_contrib', 't1_score',
          's1_stage1', 'gen_logit', 'gen_rank_pct', 'rank_u_r5b']
IDS = ['slot', 'pool', 'pair_player_lo', 'pair_player_hi', 'hand_id', 'fold', 'ev', 'm_p',
       'top5_pick', 'h_p']

d = pd.read_csv(f"{R8}/dev_pack/meta.csv")
t = np.load(f"{R8}/dev_pack/tab.npy", mmap_mode='r')
x_mom = np.nan_to_num(np.c_[t.mean(1), t.max(1)], nan=0., posinf=30., neginf=-30.).clip(-30, 30)
s = d[SCORES].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
x_base = np.c_[x_mom, s]

EXTRA = [c for c in d.columns if c not in IDS + SCORES]
x_extra = np.nan_to_num(d[EXTRA].to_numpy(float), nan=0., posinf=30., neginf=-30.).clip(-30, 30)

# ---------- board / SPR features ----------
hidx = pd.read_parquet(f"{OP}/np/hand_index.parquet")
hi_of = dict(zip(hidx.hand_id, hidx.hi))
hs = d.hand_id.map(hi_of).to_numpy()
hb = np.load(f"{OP}/np/h_board.npy", mmap_mode='r')
hbb = np.load(f"{OP}/np/h_bb.npy", mmap_mode='r')
hpot = np.load(f"{OP}/np/h_pot.npy", mmap_mode='r')
a_off = np.load(f"{OP}/np/a_off.npy", mmap_mode='r')
a_amt_to = np.load(f"{OP}/np/a_amount_to.npy", mmap_mode='r')
a_act = np.load(f"{OP}/np/a_act.npy", mmap_mode='r')
a_act_p = np.load(f"{OP}/np/a_players_active.npy", mmap_mode='r')

board = np.zeros((len(d), 8), np.float32)
for r_, hi in enumerate(hs):
    cards = hb[hi][hb[hi] >= 0]
    nb = len(cards)
    suits = cards % 4
    ranks = cards // 4
    fl = suits[:3]
    mono = 1.0 if nb >= 3 and len(set(fl)) == 1 else 0.0
    two = 1.0 if nb >= 3 and len(set(fl)) == 2 else 0.0
    paired = 1.0 if len(set(ranks.tolist())) < nb else 0.0
    fl_paired = 1.0 if len(set(ranks[:3].tolist())) < len(ranks[:3]) else 0.0
    bb = float(hbb[hi]) if hbb[hi] > 0 else 1.0
    pot_bb = float(hpot[hi]) / bb
    lo, hiA = a_off[hi], a_off[hi + 1]
    max_to_bb = float(a_amt_to[lo:hiA].max()) / bb if hiA > lo else 0.0
    n_act = float(hiA - lo)
    board[r_] = [nb, mono, two, paired, fl_paired, pot_bb, max_to_bb, n_act]
x_board = board

# ---------- set-consistency features ----------
det = np.nan_to_num(d[EXTRA].to_numpy(float), nan=0., posinf=0., neginf=0.)
mu, sd = det.mean(0), det.std(0) + 1e-9
zdet = np.clip((det - mu) / sd, -6, 6)
cal = LogisticRegression(C=10, max_iter=1000).fit(d.u_r5b.to_numpy()[:, None], d.ev.to_numpy(int))
p_cal = expit(cal.coef_[0, 0] * d.u_r5b.to_numpy() + cal.intercept_[0])

cons = np.zeros((len(d), 7), np.float32)
for slot, g in d.groupby("slot", sort=False):
    idx = g.index.to_numpy()
    X = zdet[idx]
    n = len(idx)
    sim = X @ X.T / (np.linalg.norm(X, axis=1)[:, None] * np.linalg.norm(X, axis=1)[None, :] + 1e-9)
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-g.u_r5b.to_numpy())
    pj = p_cal[idx]
    for r_ in range(n):
        v = sim[r_]
        nz = v.copy()
        top1 = np.max(nz)
        top3 = np.mean(np.sort(nz)[-3:])
        allm = np.mean(nz)
        bot5 = np.mean(np.sort(nz)[:5])
        w = expit(pj)
        wv = np.where(np.isfinite(nz), np.clip(nz, -1e3, 1e3), 0)
        wsum = np.sum(w * np.where(np.isfinite(nz), 1, 0))
        cw = float(np.sum(w * wv) / (wsum + 1e-9))
        cons[idx[r_]] = [top1, top3, allm, bot5, cw, float(np.sum(expit(pj) * wv) / (np.sum(expit(pj)) + 1e-9)), 0.0]
# rank of c_top3 within slot
c3 = pd.Series(cons[:, 1], index=d.index).groupby(d.slot).rank(pct=True).to_numpy()
cons[:, 6] = c3
CONS_COLS = ["c_top1", "c_top3", "c_all", "c_bot5", "c_w_pj", "c_w_pcal", "c_rank3"]

folds = sorted(d.fold.unique())


def run_cat(x, tag):
    y = d.ev.to_numpy(int)
    b = d.u_r5b.to_numpy()
    cal2 = LogisticRegression(C=10, max_iter=1000).fit(b[:, None], y)
    sl, ic = float(cal2.coef_[0, 0]), float(cal2.intercept_[0])
    delta = np.full(len(d), np.nan)
    for f in folds:
        va = d.fold.to_numpy() == f
        m = CatBoostClassifier(iterations=400, depth=4, learning_rate=.03, l2_leaf_reg=30,
                               random_seed=71, thread_count=16, verbose=False, allow_writing_files=False)
        m.fit(Pool(x[~va], y[~va], baseline=sl * b[~va] + ic, weight=1 / d.m_p.to_numpy()[~va]))
        delta[va] = m.predict(x[va], prediction_type='RawFormulaVal') / sl
    z = b + 0.25 * delta
    pp = per_pair(d, z)
    E = float(pp.E.mean())
    print(f"[{tag}] E={E:.6f}  dim={x.shape[1]}", flush=True)
    return z, E


def run_ranker(x, tag):
    import xgboost as xgb
    y = d.ev.to_numpy(int)
    order = np.argsort(d.slot.to_numpy(), kind='stable')
    dsort = d.iloc[order]
    xsort = x[order]
    groups = dsort.groupby('slot', sort=False).size().to_numpy()
    z = np.full(len(d), np.nan)
    for f in folds:
        tr = dsort.fold.to_numpy() != f
        va = dsort.fold.to_numpy() == f
        gtr = dsort[tr].groupby('slot', sort=False).size().to_numpy()
        m = xgb.XGBRanker(objective='rank:pairwise', n_estimators=400, learning_rate=0.04,
                          max_depth=4, subsample=0.8, colsample_bytree=0.5, reg_lambda=2.0,
                          random_state=71, n_jobs=16)
        m.fit(xsort[tr], y[tr], group=gtr)
        z[order[va]] = m.predict(xsort[va])
    pp = per_pair(d, z)
    E = float(pp.E.mean())
    print(f"[{tag}] E={E:.6f}  dim={x.shape[1]}", flush=True)
    return z, E


res = {}
_, res['A'] = run_cat(x_base, 'A base')
_, res['B'] = run_cat(np.c_[x_base, x_extra], 'B +narrow')
xB = np.c_[x_base, x_extra]
_, res['C'] = run_cat(np.c_[xB, x_board], 'C +board')
_, res['D'] = run_cat(np.c_[xB, cons], 'D +cons')
xE = np.c_[xB, x_board, cons]
zE, res['E'] = run_cat(xE, 'E +board+cons')
zF, res['F'] = run_ranker(xE, 'F xgbranker')
best = (None, -1)
for w in [0.1, 0.2, 0.3, 0.5]:
    zG = 0.5 * (zE / (np.nanstd(zE) + 1e-9)) + w * (zF / (np.nanstd(zF) + 1e-9))
    pp = per_pair(d, zG)
    if pp.E.mean() > best[1]:
        best = (w, float(pp.E.mean()))
print(f"[G blend] best w={best[0]} E={best[1]:.6f}", flush=True)
res['G'] = best[1]
res['G_w'] = best[0]

json.dump(res, open(OUT / 'r10_e1.json', 'w'), indent=2)
np.save(OUT / 'zE.npy', zE)
np.save(OUT / 'zF.npy', zF)
pd.DataFrame({'slot': d.slot, 'hand_id': d.hand_id, 'ev': d.ev, 'm_p': d.m_p,
              'zE': zE, 'zF': zF}).to_parquet(OUT / 'scores.parquet')
print(json.dumps(res, indent=2))
