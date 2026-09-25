"""R5-E13: CI's deployed patch is the UNTYPED R18 event model passed through
first_k_marginal (a Poisson-binomial over 'is this hand among the first five events'),
then stacked with TabICL at beta=3. That decoder is scale-sensitive in exactly the same
way the two-type decoder turned out to be for DT (+0.0099 from p_B x 1.5). Never tested."""
import numpy as np, pandas as pd, sys, os, json
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18')
import ci_censored_event as C
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'
lg=lambda p: np.log(np.clip(p,1e-5,1-1e-5)/(1-np.clip(p,1e-5,1-1e-5)))
FAM=os.environ.get('FAM','coordinated_isolation')
alls=C.prepare(pd.read_parquet(f'{O}/t5_dev_seq.parquet'))
pools=np.array(sorted(alls.pool.unique())); s=alls[alls.fam==FAM].reset_index(drop=True)
cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev']],on=['slot','h'],validate='one_to_one').reset_index(drop=True)
hmap=pd.read_parquet(f'{O}/np/hand_index.parquet').set_index('hand_id').hi
t45=pd.read_parquet(f'{O}/r3/t45_known_e_rerank.parquet'); t45['h']=t45.hand_id.map(hmap)
c=c.merge(t45[['slot','h','tab']],on=['slot','h'],how='left')
counts=s.groupby('slot').ev.sum(); include=C.uncensored_training_rows(s)
idx=np.array(counts.index); pool=idx//900
fmi=pd.MultiIndex.from_arrays([s.slot,s.h]); mi=pd.MultiIndex.from_arrays([c.slot,c.h]); to_c=lambda v: pd.Series(v,index=fmi).reindex(mi).values
TAB=lg(c.tab.values); R15=-c.r.values.astype(float)
P={}
for seed in (260919,11,29):
    p=np.zeros(len(s))
    for _,vi in GroupKFold(5,shuffle=True,random_state=seed).split(pools,groups=pools):
        vp=pools[vi]; tr=(~s.pool.isin(vp)).to_numpy()&include; va=s.pool.isin(vp).to_numpy()
        m=lgb.LGBMClassifier(**{**C.PARAMS,'n_jobs':6}).fit(s.loc[tr,C.FEATURES].astype(float),s.loc[tr,'ev'])
        p[va]=m.predict_proba(s.loc[va,C.FEATURES].astype(float))[:,1]
    P[seed]=p
print(f'{FAM}: sum(p) per pair median {pd.Series(P[260919]).groupby(s.slot.values).sum().median():.2f}  true evidence per pair {counts.mean():.2f}')
print(f'{"g":>5s} ' + ' '.join(f'b={b}' for b in (1.0,2.0,3.0,4.0)) + '    R15 ' + f'{C.pair_ap(c,R15,counts).mean():.4f}')
best=None
for g in (0.5,0.7,1.0,1.3,1.6,2.0,3.0,5.0):
    row=[]
    for b in (1.0,2.0,3.0,4.0):
        aps=[]
        for seed in P:
            q=C.first_k_marginal(s,np.clip(P[seed]*g,0,1)); v=to_c(q)
            aps.append(C.pair_ap(c,TAB+b*lg(v),counts).reindex(idx))
        m=float(np.mean([a.mean() for a in aps])); row.append(m)
        if best is None or m>best[0]: best=(m,g,b,aps)
    print(f'{g:5.2f} ' + ' '.join(f'{v:.4f}' for v in row),flush=True)
m,g,b,aps=best
aps1=[]
for seed in P:
    q=C.first_k_marginal(s,np.clip(P[seed],0,1)); aps1.append(C.pair_ap(c,TAB+3.0*lg(to_c(q)),counts).reindex(idx))
d=np.mean([aps[i].values-aps1[i].values for i in range(len(aps))],axis=0)
pm=pd.Series(d,index=pool).groupby(level=0).mean(); rng=np.random.default_rng(7)
bs=np.array([pm.values[rng.integers(0,len(pm),len(pm))].mean() for _ in range(4000)])
print(f'best g={g} beta={b}: {m:.4f} vs deployed-form (g=1,beta=3) {np.mean([a.mean() for a in aps1]):.4f}  d={d.mean():+.4f} poolP={float((bs>0).mean()):.3f}')
