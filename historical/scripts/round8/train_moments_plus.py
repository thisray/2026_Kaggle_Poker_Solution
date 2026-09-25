"""Round-8 combo: decision moments + 11 score features residual (CPU, 5-fold, seed 71)."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
import sys
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/round8-research-20260917/code")
from metrics import per_pair

SCORES = ['sc_r5','u0','u_rr','u_r5b','lin_contrib','nn_contrib','t1_score','s1_stage1','gen_logit','gen_rank_pct','rank_u_r5b']

def matrix(root, meta):
    t = np.load(Path(root)/'tab.npy', mmap_mode='r', allow_pickle=False)
    x = np.nan_to_num(np.c_[t.mean(1), t.max(1)], nan=0., posinf=30., neginf=-30.).clip(-30, 30)
    s = meta[SCORES].replace([np.inf,-np.inf], np.nan).fillna(0).to_numpy(float)
    return np.c_[x, s]

def fit(x, d, out, seed, threads):
    b = d.u_r5b.to_numpy(); y = d.ev.to_numpy(int)
    cal = LogisticRegression(C=10, max_iter=1000).fit(b[:,None], y); s = float(cal.coef_[0,0]); c = float(cal.intercept_[0])
    m = CatBoostClassifier(iterations=400, depth=4, learning_rate=.03, l2_leaf_reg=30,
        random_seed=seed, thread_count=threads, verbose=False, allow_writing_files=False)
    m.fit(Pool(x, y, baseline=s*b+c, weight=1/d.m_p.to_numpy()))
    out = Path(out); out.mkdir(parents=True, exist_ok=True); m.save_model(str(out/'model.cbm'))
    (out/'model.json').write_text(json.dumps({'slope': s, 'scale': .25, 'seed': seed, 'feature_dim': x.shape[1]}))
    return m, s

def main():
    p = argparse.ArgumentParser(); p.add_argument('--pack', required=True); p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=71); p.add_argument('--threads', type=int, default=4); p.add_argument('--scale', type=float, default=.25)
    a = p.parse_args()
    r = Path(a.pack); d = pd.read_csv(r/'meta.csv'); x = matrix(r, d); o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    delta = np.full(len(d), np.nan); z = np.full(len(d), np.nan)
    for f in sorted(d.fold.unique()):
        va = d.fold.to_numpy() == f
        m, s = fit(x[~va], d.loc[~va], o/f'fold{f}', a.seed, a.threads)
        delta[va] = m.predict(x[va], prediction_type='RawFormulaVal')/s
        z[va] = d.u_r5b.to_numpy()[va] + a.scale*delta[va]
        print('completed fold', int(f), flush=True)
    keep = np.isfinite(z); sub = d.loc[keep].reset_index(drop=True)
    base = per_pair(sub, sub.u_r5b.to_numpy()); new = per_pair(sub, z[keep]); new.to_csv(o/'per_pair.csv', index=False)
    res = {'baseline_E': float(base.E.mean()), 'E': float(new.E.mean()), 'delta_E': float(new.E.mean()-base.E.mean()),
           'scale': a.scale, 'features': '588 moments + 11 scores', 'evaluated_pairs': len(new)}
    (o/'cv.json').write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
