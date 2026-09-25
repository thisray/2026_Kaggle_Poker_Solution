"""R5-E4: is the remaining evidence error irreducible, or just badly combined?
(a) per-pair oracle over the zoo's slates and the union recall of their top-5 sets:
    if the models made the same (Bayes-limited) errors these would sit at the mean;
(b) a LEARNED candidate-level combiner (the P lane deliberately avoided fitted
    weights, but there the ceiling was 0.02 away - here it is 0.24), pool-grouped CV
    so no pair's pool trains its own combiner."""
import numpy as np, pandas as pd, sys, os, json, glob, lightgbm as lgb
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18')
import ci_censored_event as C
from sklearn.model_selection import GroupKFold
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; R5=f'{O}/r5'
FAM=os.environ['FAM']; ab=FAM[:2]
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9)))
s=pd.read_parquet(f'{R5}/L/rows_{ab}.parquet').reset_index(drop=True); counts=s.groupby('slot').ev.sum()
cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
hmap=pd.read_parquet(f'{O}/np/hand_index.parquet').set_index('hand_id').hi
t45=pd.read_parquet(f'{O}/r3/t45_known_e_rerank.parquet'); t45['h']=t45.hand_id.map(hmap)
extra=[c for c in t45.columns if c in ('tab','rs_blend','sur_c_max','sur_c_mean','sur_c_aggr_max','sur_1_max','sur_1_aggr_max','nd','nag','gap_aggr','gap_max','p_base_feats_only','p_with_memo_feats')]
c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev','pool']],on=['slot','h'],validate='one_to_one').merge(t45[['slot','h']+extra],on=['slot','h'],how='left').reset_index(drop=True)
mi=pd.MultiIndex.from_arrays([c.slot,c.h]); fmi=pd.MultiIndex.from_arrays([s.slot,s.h])
to_c=lambda v: pd.Series(v,index=fmi).reindex(mi).values
files=sorted(glob.glob(f'{R5}/L/{ab}__*.npy')); names=[os.path.basename(f).split('__')[1][:-4] for f in files]
Z={n:np.load(f) for n,f in zip(names,files)}; NS=Z[names[0]].shape[0]
ap=lambda v: C.pair_ap(c,v,counts)
print(f'=== {FAM}: {len(names)} configs, {len(counts)} pairs')
# ---------- (a) diversity ----------
per={n:[ap(to_c(Z[n][si][2])) for si in range(NS)] for n in names}
mean_of=lambda n: float(np.mean([x.mean() for x in per[n]]))
M=pd.DataFrame({n:np.mean([per[n][si].values for si in range(NS)],axis=0) for n in names},index=counts.index)
print(f'   single-config mean AP@5: min {M.mean().min():.4f} max {M.mean().max():.4f}')
print(f'   per-pair ORACLE over configs: {M.max(axis=1).mean():.4f}   (mean of configs {M.mean(axis=1).mean():.4f})')
# union recall of top-5 sets, seed 0
top5={}
for n in names:
    v=to_c(Z[n][0][2]); z=c[['slot','h','ev']].copy(); z['s']=v
    z=z.sort_values(['slot','s'],ascending=[True,False]); z['r']=z.groupby('slot').cumcount()+1
    top5[n]=set(map(tuple,z[z.r<=5][['slot','h']].values))
tru=set(map(tuple,c[c.ev.astype(bool)][['slot','h']].values))
uni=set().union(*top5.values())
print(f'   true evidence in candidates {len(tru)}; single top-5 hit {np.mean([len(tru&top5[n]) for n in names]):.1f}; UNION of all {len(names)} top-5 sets covers {len(tru&uni)} ({len(tru&uni)/len(tru):.3f}); union size {len(uni)/len(counts):.1f} per pair')
# ---------- (b) learned combiner ----------
F=pd.DataFrame({f'L_{n}':lg(to_c(np.mean([Z[n][si][2] for si in range(NS)],axis=0))) for n in names})
F['pA_mean']=lg(to_c(np.mean([Z[n][si][0] for n in names for si in range(NS)],axis=0)))
F['pB_mean']=lg(to_c(np.mean([Z[n][si][1] for n in names for si in range(NS)],axis=0)))
F['r15']=-c.r.values
for e in extra: F[e]=c[e].values
F['ts_rel']=c.groupby('slot').ts.rank(pct=True).values
pools=np.array(sorted(c.pool.unique())); y=c.ev.astype(int).values; oof=np.zeros(len(c))
for seed in (11,29,260919):
    o=np.zeros(len(c))
    for _,vi in GroupKFold(5,shuffle=True,random_state=seed).split(pools,groups=pools):
        vp=pools[vi]; tr=(~c.pool.isin(vp)).to_numpy(); va=~tr
        m=lgb.LGBMClassifier(n_estimators=350,learning_rate=.04,num_leaves=15,min_child_samples=40,colsample_bytree=.7,subsample=.8,subsample_freq=1,reg_lambda=10,verbosity=-1,n_jobs=4,random_state=seed).fit(F[tr],y[tr])
        o[va]=m.predict_proba(F[va])[:,1]
    oof+=o/3
print(f'   learned combiner (pool CV x3 seeds): {ap(oof).mean():.4f}')
best=max(names,key=mean_of); print(f'   best single config: {best} {mean_of(best):.4f} | R15 {ap(-c.r.values).mean():.4f} | tabicl {ap(lg(c.tab.values)).mean():.4f}')
for w in (0.3,0.5,0.7):
    blend=w*pd.Series(oof).groupby(c.slot.values).rank(pct=True).values+(1-w)*pd.Series(lg(c.tab.values)).groupby(c.slot.values).rank(pct=True).values
    print(f'   combiner rank-blend w={w} with tabicl: {ap(blend).mean():.4f}')
json.dump({'oracle_over_configs':float(M.max(axis=1).mean()),'mean_of_configs':float(M.mean(axis=1).mean()),'union_recall':len(tru&uni)/len(tru),'learned_combiner':float(ap(oof).mean())},open(f'{R5}/e4_{ab}.json','w'),indent=1)
