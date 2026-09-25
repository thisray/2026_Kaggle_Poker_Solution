"""R5-P1b: R3 tested pool CENTERING (alpha=-1) and it was significantly negative, which
was read as 'pool-level offsets carry signal'. The symmetric question was never asked:
does AMPLIFYING the pool offset help? Positives are ~Poisson(0.93) per pool, so under
homogeneity pairs are independent and neither should help - a clean discriminative test.
score' = logit(p) + alpha * (pool_mean_logit - global_mean_logit)"""
import numpy as np, pandas as pd, sys, json
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker/src')
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917')
from pokerlab.metrics import stable_ap
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'
lg=lambda p: np.log(np.clip(p,1e-7,1-1e-7)/(1-np.clip(p,1e-7,1-1e-7)))
import glob, os
cands=['m15_o_pos_a_train_oof.parquet','m15_o_goss_b_train_oof.parquet','m5_both_train_oof.parquet']
src=[f for f in cands if os.path.exists(f'{O}/{f}')]
print('using',src)
out={}
for f in src:
    T=pd.read_parquet(f'{O}/{f}')
    for SUB in ('devsub11','devsub12'):
        d=T[T.src==SUB].copy()
        hid=set(T[(T.src==SUB)&(T.label==-1)&(T.oof>0.3)].key)
        d=d[~d.key.isin(hid)].sort_values('key',kind='mergesort').reset_index(drop=True)
        if 'slot' in d.columns: pool=(d.slot//900).values
        else:
            pl=pd.read_parquet(f'{O}/player_local_v1.parquet').set_index('player_gi')
            lo=(d.key//12000).values; pool=pl.pool.reindex(lo).values
        y=(d.y.values==1).astype(int); z=lg(d.oof.values)
        pm=pd.Series(z).groupby(pool).transform('mean').values; gm=z.mean()
        row={}
        for a in (-1.0,-0.5,-0.25,-0.1,0.0,0.1,0.25,0.5,1.0):
            row[a]=round(float(stable_ap(y,z+a*(pm-gm))),5)
        out[f'{f[:22]}|{SUB}']=row
        print(f'{f[:22]:24s} {SUB}: '+' '.join(f'a{a:+.2f}:{v:.5f}' for a,v in row.items()),flush=True)
json.dump(out,open(f'{O}/r5/p1_pooloffset.json','w'),indent=1)
