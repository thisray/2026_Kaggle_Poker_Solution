"""P-loss anatomy on devsub OOF: where does AP (full population) lose, by family / exposure / rank."""
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as aps
A = '/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/'
for name in ['m15_v6ens_base', 'm15_v6ens_cat']:
    d = pd.read_parquet(A + name + '_train_oof.parquet')
    print('==', name, d.shape, d.src.value_counts().to_dict())
    for src, g in d.groupby('src'):
        g = g.copy()
        lab = g[g.label >= 0]
        ap_clean = aps(lab.y, lab.oof)
        # full population: U treated as negative, hidden-positive suspects removed
        nh = g[~g.hid]
        ap_nohid = aps(nh.y, nh.oof)
        ap_raw = aps(g.y, g.oof)
        npos = int(g.y.sum())
        print(f'  {src}: n={len(g)} pos={npos} hid={int(g.hid.sum())} AP_clean={ap_clean:.4f} AP_nohid={ap_nohid:.4f} AP_raw={ap_raw:.4f}')
        nh = nh.sort_values('oof', ascending=False).reset_index(drop=True)
        nh['rank'] = np.arange(1, len(nh) + 1)
        pos = nh[nh.y == 1].copy()
        pos['cumpos'] = np.arange(1, len(pos) + 1)
        pos['prec'] = pos.cumpos / pos['rank']
        # AP loss contributions: (1 - prec)/npos
        pos['loss'] = (1 - pos.prec) / len(pos)
        print('    rank quantiles of positives:', np.percentile(pos['rank'], [50, 75, 90, 95, 98, 100]).round(0).tolist())
        bins = [0, 100, 200, 300, 400, 500, 750, 1000, 2000, 5000, 1e9]
        pos['rb'] = pd.cut(pos['rank'], bins)
        print('    positives by rank bin / AP loss share:')
        print(pos.groupby('rb', observed=True).agg(n=('y', 'size'), loss=('loss', 'sum')).to_string())
        print('    by family: n, mean rank, loss')
        print(pos.groupby('fam').agg(n=('y', 'size'), med_rank=('rank', 'median'), loss=('loss', 'sum')).to_string())
        pos['nb'] = pd.qcut(pos.n, 4)
        print(pos.groupby('nb', observed=True).agg(n=('y', 'size'), med_rank=('rank', 'median'), loss=('loss', 'sum')).to_string())
        # negatives above rank 400 composition
        top = nh.head(450)
        print('    top-450 composition:', top.y.sum(), 'pos;', (top.label == 0).sum(), 'confirmed neg;', ((top.label < 0) & (top.y == 0)).sum(), 'U')
