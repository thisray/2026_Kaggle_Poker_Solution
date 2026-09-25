"""R5-E17: soft_play's A/B boundary is fuzzy (single-feature AUC 0.69-0.76 vs DT's 0.94),
and its TYPED decoder alone is 0.676 - WORSE than R15's 0.720. Every SP experiment so far
has assumed the two-type decoder. Test the untyped alternative that CI uses: one event
model over all listed hands, first_k_marginal, then the deployed blends. A fusion of
untyped configs is also tested."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, os, json
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18')
import ci_censored_event as C
from sklearn.model_selection import GroupKFold
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; R5=f'{O}/r5'
FAM=os.environ.get('FAM','soft_play'); NJ=int(os.environ.get('NJ',4))
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9))); sg=lambda x:1/(1+np.exp(-x))
base=C.prepare(pd.read_parquet(f'{O}/t5_dev_seq.parquet'))
nf=pd.read_parquet(f'{O}/r3/t58_seq_feats.parquet'); NEW=[c for c in nf.columns if c not in ('slot','h','pa','pb')]
base=base.merge(nf[['slot','h']+NEW],on=['slot','h'],how='left'); base[NEW]=base[NEW].fillna(0.0)
role=pd.read_parquet(f'{O}/r4/x2c_role_dev.parquet'); ROLE=[c for c in role.columns if c.startswith('x_') and c not in ('x_k','x_n')]
ker=pd.read_parquet(f'{O}/r4/x11_kernel_dev.parquet'); KER=[c for c in ker.columns if c.startswith('k_')]
full=base.merge(role[['slot','h']+ROLE],on=['slot','h']).merge(ker[['slot','h']+KER],on=['slot','h']).sort_values(['slot','ts','h'],kind='stable').reset_index(drop=True)
FS=C.FEATURES+NEW+ROLE+KER
s=full[full.fam==FAM].reset_index(drop=True); counts=s.groupby('slot').ev.sum(); include=C.uncensored_training_rows(s)
cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
hmap=pd.read_parquet(f'{O}/np/hand_index.parquet').set_index('hand_id').hi
t45=pd.read_parquet(f'{O}/r3/t45_known_e_rerank.parquet'); t45['h']=t45.hand_id.map(hmap)
c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev']],on=['slot','h'],validate='one_to_one').merge(t45[['slot','h','tab']],on=['slot','h'],how='left').reset_index(drop=True)
mi=pd.MultiIndex.from_arrays([c.slot,c.h]); fmi=pd.MultiIndex.from_arrays([s.slot,s.h]); to_c=lambda v: pd.Series(v,index=fmi).reindex(mi).values
rk=lambda v: pd.Series(v).groupby(c.slot.values).rank(pct=True).values
TAB=lg(c.tab.values); R15=-c.r.values.astype(float); idx=np.array(counts.index); pool=idx//900
pools=np.array(sorted(full.pool.unique()))
CFG={'base':dict(C.PARAMS),'deep':{**C.PARAMS,'num_leaves':63,'min_child_samples':20,'n_estimators':500,'learning_rate':.03},
     'shal':{**C.PARAMS,'num_leaves':7,'n_estimators':900,'learning_rate':.02,'min_child_samples':60},
     'goss':{**C.PARAMS,'boosting_type':'goss','num_leaves':31,'n_estimators':500},
     'extra':{**C.PARAMS,'extra_trees':True,'num_leaves':31,'n_estimators':600},
     'col3':{**C.PARAMS,'colsample_bytree':.3,'num_leaves':31,'n_estimators':600},
     'l2':{**C.PARAMS,'reg_lambda':200,'num_leaves':31,'n_estimators':600}}
SEEDS=(260919,11,29); Ps={}
for nm,par in CFG.items():
    for seed in SEEDS:
        p=np.zeros(len(s))
        for _,vi in GroupKFold(5,shuffle=True,random_state=seed).split(pools,groups=pools):
            vp=pools[vi]; tr=(~s.pool.isin(vp)).to_numpy()&include; va=s.pool.isin(vp).to_numpy()
            p[va]=lgb.LGBMClassifier(**{**par,'n_jobs':NJ,'random_state':seed}).fit(s.loc[tr,FS].astype(float),s.loc[tr,'ev']).predict_proba(s.loc[va,FS].astype(float))[:,1]
        Ps[(nm,seed)]=p
    print(f'  {nm} done',flush=True)
print(f'{FAM}: sum(p) per pair median {pd.Series(Ps[("base",SEEDS[0])]).groupby(s.slot.values).sum().median():.2f}  true evidence/pair {counts.mean():.2f}')
def ev(name,qs):
    out={}
    for fn,f in (('alone',lambda q: q),('stack1',lambda q: TAB+1.0*lg(q)),('stack2',lambda q: TAB+2.0*lg(q)),('stack3',lambda q: TAB+3.0*lg(q)),
                 ('rank0.35',lambda q: 0.65*rk(R15)+0.35*rk(q)),('rank0.5',lambda q: 0.5*rk(R15)+0.5*rk(q)),('rank0.65',lambda q: 0.35*rk(R15)+0.65*rk(q))):
        aps=[C.pair_ap(c,f(to_c(q)),counts).reindex(idx) for q in qs]
        out[fn]=(float(np.mean([a.mean() for a in aps])),aps)
    for k,(m,_) in sorted(out.items(),key=lambda kv:-kv[1][0])[:3]: print(f'  {name:26s} {k:9s} {m:.4f}',flush=True)
    return out
R={}
for g in (0.7,1.0,1.5,2.0):
    R[f'fuse7_g{g}']=ev(f'untyped fuse7 g={g}',[C.first_k_marginal(s,np.clip(sg(np.mean([lg(Ps[(nm,seed)]) for nm in CFG],axis=0))*g,0,1)) for seed in SEEDS])
R['base_g1']=ev('untyped base g=1',[C.first_k_marginal(s,np.clip(Ps[('base',seed)],0,1)) for seed in SEEDS])
print(f'R15 reference: {C.pair_ap(c,R15,counts).mean():.4f}')
rng=np.random.default_rng(7); ref=[C.pair_ap(c,R15,counts).reindex(idx)]*len(SEEDS)
best=max(((k,f,v[0],v[1]) for k,d in R.items() for f,v in d.items()),key=lambda t:t[2])
d=np.mean([best[3][i].values-ref[i].values for i in range(len(SEEDS))],axis=0)
pm=pd.Series(d,index=pool).groupby(level=0).mean(); bs=np.array([pm.values[rng.integers(0,len(pm),len(pm))].mean() for _ in range(4000)])
print(f'BEST untyped: {best[0]}|{best[1]} = {best[2]:.4f}  vs R15 d={d.mean():+.4f} poolP={float((bs>0).mean()):.3f}')
