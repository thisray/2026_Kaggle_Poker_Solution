"""Round-8 deployment: fit combo residual on all dev; predict per eval pack chunk."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
import sys
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/round8-research-20260917/code")

SCORES = ['sc_r5','u0','u_rr','u_r5b','lin_contrib','nn_contrib','t1_score','s1_stage1','gen_logit','gen_rank_pct','rank_u_r5b']

def matrix(root, meta):
    t = np.load(Path(root)/'tab.npy', mmap_mode='r', allow_pickle=False)
    x = np.nan_to_num(np.c_[t.mean(1), t.max(1)], nan=0., posinf=30., neginf=-30.).clip(-30, 30)
    s = meta[SCORES].replace([np.inf,-np.inf], np.nan).fillna(0).to_numpy(float)
    return np.c_[x, s]

def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['fit','predict'])
    p.add_argument('--pack'); p.add_argument('--out'); p.add_argument('--model-dir')
    p.add_argument('--seed', type=int, default=71); p.add_argument('--threads', type=int, default=8); p.add_argument('--scale', type=float, default=.25)
    a = p.parse_args()
    if a.mode == 'fit':
        r = Path(a.pack); d = pd.read_csv(r/'meta.csv'); x = matrix(r, d)
        b = d.u_r5b.to_numpy(); y = d.ev.to_numpy(int)
        cal = LogisticRegression(C=10, max_iter=1000).fit(b[:,None], y); s = float(cal.coef_[0,0]); c = float(cal.intercept_[0])
        m = CatBoostClassifier(iterations=400, depth=4, learning_rate=.03, l2_leaf_reg=30,
            random_seed=a.seed, thread_count=a.threads, verbose=False, allow_writing_files=False)
        m.fit(Pool(x, y, baseline=s*b+c, weight=1/d.m_p.to_numpy()))
        o = Path(a.out); o.mkdir(parents=True, exist_ok=True); m.save_model(str(o/'model.cbm'))
        (o/'model.json').write_text(json.dumps({'slope': s, 'scale': a.scale, 'seed': a.seed, 'features': 'moments+scores'}))
        print('fitted', x.shape, 'slope', round(s,4))
    else:
        mr = Path(a.model_dir); info = json.loads((mr/'model.json').read_text())
        m = CatBoostClassifier(); m.load_model(str(mr/'model.cbm'))
        r = Path(a.pack); d = pd.read_csv(r/'meta.csv'); x = matrix(r, d)
        delta = m.predict(x, prediction_type='RawFormulaVal')/info['slope']
        z = d.u_r5b.to_numpy() + info['scale']*delta
        out = d[['slot','pool','hand_id']].copy(); out['base'] = d.u_r5b; out['delta'] = delta; out['score'] = z
        out.to_csv(a.out, index=False); print('scored', len(out), '->', a.out)

if __name__ == '__main__':
    main()
