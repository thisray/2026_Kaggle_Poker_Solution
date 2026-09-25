"""Round-11 deployable ranker variant: moments + 11 scores + within-pair score consistency.

Same corrected recipe as code/ranker_bridge.py (fold-internal calibration, no bagging,
fixed .4/.6 rank blend) but without the gameplay 'extras' that are unavailable for the
evaluation candidate table. Consistency features are computed on the 11 scores.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata

SCORES = ['sc_r5', 'u0', 'u_rr', 'u_r5b', 'lin_contrib', 'nn_contrib', 't1_score',
          's1_stage1', 'gen_logit', 'gen_rank_pct', 'rank_u_r5b']


def load_pack(pack):
    pack = Path(pack); d = pd.read_csv(pack / 'meta.csv')
    if d.duplicated(['slot', 'hand_id']).any(): raise ValueError('Duplicate candidate keys')
    if (pack / 'moments.npy').exists(): x = np.load(pack / 'moments.npy', mmap_mode='r')
    else:
        t = np.load(pack / 'tab.npy', mmap_mode='r'); x = np.c_[t.mean(1), t.max(1)]
    if len(x) != len(d) or x.ndim != 2: raise ValueError('Pack alignment mismatch')
    x = np.nan_to_num(x, nan=0., posinf=30., neginf=-30.).clip(-30, 30).astype(np.float32)
    return d.reset_index(drop=True), x


def dense(d, columns):
    if set(columns) - set(d): raise ValueError(f'Missing columns: {sorted(set(columns)-set(d))}')
    return np.nan_to_num(d[columns].to_numpy(float), nan=0., posinf=30., neginf=-30.).clip(-30, 30)


def consistency(d, raw, mu, sd):
    z = np.clip((raw - mu) / sd, -6, 6); out = np.zeros((len(d), 3))
    for ix in d.groupby('slot', sort=False).indices.values():
        y = z[ix]; norm = np.linalg.norm(y, axis=1)
        sim = y @ y.T / (norm[:, None] * norm[None, :] + 1e-9)
        for local, row in enumerate(ix):
            others = np.delete(sim[local], local)
            if len(others):
                ordered = np.sort(others)
                out[row] = [ordered[-min(3, len(ordered)):].mean(), ordered[-1], ordered[:min(5, len(ordered))].mean()]
    if not np.isfinite(out).all(): raise AssertionError('Invalid consistency feature')
    return out


def build_x(d, moments, mu, sd):
    raw = dense(d, SCORES)
    return np.c_[moments, raw, consistency(d, raw, mu, sd)].astype(np.float32)


def pair_ranks(d, scores):
    out = np.empty(len(d))
    for ix in d.groupby('slot', sort=False).indices.values():
        out[ix] = rankdata(-np.asarray(scores)[ix], method='average')
    return out


def metrics(d, scores):
    rows = []
    for slot, ix in d.groupby('slot', sort=False).indices.items():
        g = d.iloc[ix]; y = g.ev.to_numpy(int); m = int(g.m_p.iloc[0])
        if m < 1 or (g.m_p != m).any(): raise ValueError('Full truth count required')
        order = np.argsort(-np.asarray(scores)[ix], kind='stable'); top = y[order[:5]]
        e = float(np.sum(top * np.cumsum(top) / np.arange(1, len(top) + 1)) / min(m, 5))
        rows.append({'slot': slot, 'pool': g.pool.iloc[0], 'fold': g.fold.iloc[0], 'E': e,
                     'hits5': int(top.sum()), 'm_p': m})
    return pd.DataFrame(rows)


def fit_one(d, moments, folder, seed=71, rounds=500, cat_rounds=400, threads=4):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    raw = dense(d, SCORES); mu = raw.mean(0); sd = raw.std(0) + 1e-9
    x = build_x(d, moments, mu, sd); y = d.ev.to_numpy(int); b = d.u_r5b.to_numpy(float)
    cal = LogisticRegression(C=10, max_iter=1000).fit(b[:, None], y)
    slope = float(cal.coef_[0, 0]); intercept = float(cal.intercept_[0])
    if slope <= 1e-6: raise ValueError('Nonpositive baseline calibration')
    cat = CatBoostClassifier(iterations=cat_rounds, depth=4, learning_rate=.03, l2_leaf_reg=30,
                             random_seed=seed, thread_count=threads, verbose=False, allow_writing_files=False)
    cat.fit(Pool(x, y, baseline=slope * b + intercept, weight=1 / d.m_p.to_numpy(float)))
    order = np.argsort(d.slot.to_numpy(), kind='stable'); ds = d.iloc[order]
    groups = ds.groupby('slot', sort=False).size().to_numpy()
    lr = lgb.LGBMRanker(objective='lambdarank', n_estimators=rounds, learning_rate=.04, num_leaves=31,
                        min_child_samples=20, colsample_bytree=.5, reg_lambda=2.,
                        lambdarank_truncation_level=5, label_gain=[0, 1], random_state=seed,
                        n_jobs=threads, verbosity=-1, subsample=1., subsample_freq=0,
                        deterministic=True, force_col_wise=True)
    lr.fit(x[order], y[order], group=groups)
    cat.save_model(str(folder / 'cat.cbm')); lr.booster_.save_model(str(folder / 'ranker.txt'))
    np.savez(folder / 'scaling.npz', mu=mu, sd=sd)
    cfg = dict(scores=SCORES, extras=[], moment_dim=moments.shape[1], slope=slope, intercept=intercept,
               seed=seed, rank_rounds=rounds, cat_rounds=cat_rounds, cat_weight=.4,
               variant='score-consistency simplified (no gameplay extras)')
    (folder / 'model.json').write_text(json.dumps(cfg, indent=2))
    return cfg


def predict_one(d, moments, folder):
    folder = Path(folder); cfg = json.loads((folder / 'model.json').read_text())
    if moments.shape[1] != cfg['moment_dim']: raise ValueError('Moments dimension changed')
    norm = np.load(folder / 'scaling.npz'); x = build_x(d, moments, norm['mu'], norm['sd'])
    cat = CatBoostClassifier(); cat.load_model(str(folder / 'cat.cbm'))
    lr = lgb.Booster(model_file=str(folder / 'ranker.txt'))
    zc = d.u_r5b.to_numpy(float) + .25 * cat.predict(x, prediction_type='RawFormulaVal') / cfg['slope']
    zl = lr.predict(x)
    zb = -(cfg['cat_weight'] * pair_ranks(d, zc) + (1 - cfg['cat_weight']) * pair_ranks(d, zl))
    return zc, zl, zb


def fit(pack, out, seed=71, rounds=500, cat_rounds=400, threads=4, cv_only=False):
    out = Path(out)
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    d, mom = load_pack(pack)
    if {'fold', 'ev', 'm_p', 'pool'} - set(d): raise ValueError('CV requires full truth, fold and pool')
    if d.groupby('pool').fold.nunique().max() > 1: raise ValueError('Pool label leakage')
    zs = np.full((len(d), 3), np.nan)
    for fold in sorted(d.fold.unique()):
        va = d.fold.to_numpy() == fold; tr = ~va
        fit_one(d.loc[tr].reset_index(drop=True), mom[tr], out / f'fold_{fold}', seed, rounds, cat_rounds, threads)
        pred = predict_one(d.loc[va].reset_index(drop=True), mom[va], out / f'fold_{fold}')
        zs[va] = np.column_stack(pred)
    pred = d[[c for c in ['slot', 'pair_id', 'hand_id', 'pool', 'fold', 'ev', 'm_p'] if c in d]].copy()
    report = {'baseline_E': float(metrics(d, d.u_r5b).E.mean()),
              'scope': 'frozen-upstream CV; score-consistency simplified variant'}
    for j, name in enumerate(['cat', 'ranker', 'blend']):
        pred[name] = zs[:, j]; mp = metrics(d, zs[:, j]); report[name + '_E'] = float(mp.E.mean())
        mp.to_csv(out / f'{name}_per_pair.csv', index=False)
    pred['score'] = pred['blend']
    pred.to_csv(out / 'oof_scores.csv.gz', index=False)
    if not cv_only: fit_one(d, mom, out / 'full', seed, rounds, cat_rounds, threads)
    (out / 'cv_summary.json').write_text(json.dumps(report, indent=2)); return report


def main():
    p = argparse.ArgumentParser(); s = p.add_subparsers(dest='cmd', required=True)
    f = s.add_parser('fit'); f.add_argument('--pack', required=True); f.add_argument('--out', required=True)
    f.add_argument('--seed', type=int, default=71); f.add_argument('--rounds', type=int, default=500)
    f.add_argument('--cat-rounds', type=int, default=400); f.add_argument('--threads', type=int, default=4)
    f.add_argument('--cv-only', action='store_true')
    r = s.add_parser('predict'); r.add_argument('--pack', required=True); r.add_argument('--model', required=True); r.add_argument('--out', required=True)
    a = p.parse_args(); args = vars(a); cmd = args.pop('cmd')
    if cmd == 'fit': print(json.dumps(fit(**args), indent=2))
    else:
        d, m = load_pack(a.pack); cat, lr, blend = predict_one(d, m, a.model)
        out = d[[c for c in ['slot', 'pair_id', 'hand_id'] if c in d]].copy()
        out['score'] = blend; out['ranker'] = lr; out['cat'] = cat
        if Path(a.out).exists(): raise FileExistsError(a.out)
        out.to_csv(a.out, index=False)


if __name__ == '__main__': main()
