"""R5-E23: the pair model gained +0.0071 from exposure-aligned replicas (same pair, a
different 2/3 of its hands). The EVENT models never got that treatment - they train on
dev pairs with ~121 co-seated hands and are applied to eval pairs with ~86.
A valid replica for the event model keeps EVERY listed evidence hand (so the listing
structure and the censoring masks are untouched) and drops a random fraction of the
non-evidence hands; the within-pair exposure features (relative position, 1/n) are
recomputed on the replica. Training set = union of replicas; evaluation is always on the
REAL full frame, standard protocol."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, os, json
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18'); sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
from sklearn.model_selection import GroupKFold
import ci_censored_event as C
from g4_decode import decode, runs
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; R5=f'{O}/r5'; NJ=int(os.environ.get('NJ',6))
FAM=os.environ.get('FAM','directed_transfer'); GB=float(os.environ.get('GB','1.5')); KEEP=float(os.environ.get('KEEP','0.70')); NREP=int(os.environ.get('NREP','3'))
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9))); sg=lambda x:1/(1+np.exp(-x))
base=C.prepare(pd.read_parquet(f'{O}/t5_dev_seq.parquet'))
nf=pd.read_parquet(f'{O}/r3/t58_seq_feats.parquet'); NEW=[c for c in nf.columns if c not in ('slot','h','pa','pb')]
base=base.merge(nf[['slot','h']+NEW],on=['slot','h'],how='left'); base[NEW]=base[NEW].fillna(0.0)
role=pd.read_parquet(f'{O}/r4/x2c_role_dev.parquet'); ROLE=[c for c in role.columns if c.startswith('x_') and c not in ('x_k','x_n')]
ker=pd.read_parquet(f'{O}/r4/x11_kernel_dev.parquet'); KER=[c for c in ker.columns if c.startswith('k_')]
full=base.merge(role[['slot','h']+ROLE],on=['slot','h']).merge(ker[['slot','h']+KER],on=['slot','h'])
FS=C.FEATURES+NEW+ROLE+KER
g1=pd.read_parquet(f'{O}/r4/g1_evidence_rank.parquet')[['h','pair_id','evidence_rank','chron']].sort_values(['pair_id','evidence_rank'])
g1['run']=np.concatenate([runs(g.chron.values) for _,g in g1.groupby('pair_id',sort=False)]); g1['nruns']=g1.groupby('pair_id').run.transform('max')+1
full=full.merge(g1[['h','run','nruns']],on='h',how='left').sort_values(['slot','ts','h'],kind='stable').reset_index(drop=True)
s=full[full.fam==FAM].reset_index(drop=True); counts=s.groupby('slot').ev.sum()
e=s[s.ev.astype(bool)]; two=e[e.nruns==2]
clf=lgb.LGBMClassifier(n_estimators=200,learning_rate=.05,num_leaves=7,min_child_samples=10,colsample_bytree=.5,verbosity=-1,n_jobs=NJ).fit(two[FS].astype(float),(two.run==1).astype(int))
typB=(s.pa_at_trig<6).values if FAM=='coordinated_isolation' else np.where(s.nruns.values==2,s.run.values==1,clf.predict_proba(s[FS].astype(float))[:,1]>0.5)
s['isA']=s.ev.astype(bool)&~typB; s['isB']=s.ev.astype(bool)&typB
def masks(d):
    nA=d.groupby('slot').isA.transform('sum'); nB=d.groupby('slot').isB.transform('sum'); nev=d.groupby('slot').ev.transform('sum')
    lastA=d.slot.map(d[d.isA].groupby('slot').ts.max()); lastB=d.slot.map(d[d.isB].groupby('slot').ts.max())
    return ((nA<5)|(d.ts<=lastA)).values, ((nev<5)|((nB>0)&(d.ts<=lastB))).values
incA,incB=masks(s)
def replica(rs):
    rng=np.random.default_rng(rs)
    keep=s.ev.astype(bool).values | (rng.random(len(s))<KEEP)
    r=s[keep].copy().reset_index(drop=True)
    r['x_rel']=r.groupby('slot').ts.rank(pct=True).values          # recompute exposure features on the replica
    n=r.groupby('slot').ts.transform('size').values; r['x_inv_n']=1.0/np.maximum(n,1)
    a,b=masks(r); return r,a,b
cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
hmap=pd.read_parquet(f'{O}/np/hand_index.parquet').set_index('hand_id').hi
t45=pd.read_parquet(f'{O}/r3/t45_known_e_rerank.parquet'); t45['h']=t45.hand_id.map(hmap)
c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev']],on=['slot','h'],validate='one_to_one').merge(t45[['slot','h','tab']],on=['slot','h'],how='left').reset_index(drop=True)
mi=pd.MultiIndex.from_arrays([c.slot,c.h]); fmi=pd.MultiIndex.from_arrays([s.slot,s.h]); to_c=lambda v: pd.Series(v,index=fmi).reindex(mi).values
rk=lambda v: pd.Series(v).groupby(c.slot.values).rank(pct=True).values
TAB=lg(c.tab.values); R15=-c.r.values.astype(float); idx=np.array(counts.index); pool=idx//900; rng=np.random.default_rng(7)
sc=(lambda L: 0.65*rk(R15)+0.35*rk(L)) if FAM=='soft_play' else (lambda L: TAB+(3.0 if FAM=='coordinated_isolation' else 1.0)*lg(np.clip(L,1e-9,None)))
pools=np.array(sorted(full.pool.unique()))
REPS=[replica(1000+i) for i in range(NREP)]
print(f'{FAM}: full frame {len(s)} rows, mean hands/pair {len(s)/s.slot.nunique():.1f}; replica keeps {KEEP:.0%} of non-evidence -> {len(REPS[0][0])} rows, {len(REPS[0][0])/s.slot.nunique():.1f} hands/pair (eval is ~86)',flush=True)
def run(augment):
    aps=[]
    for seed in (260919,11,29):
        pA=np.zeros(len(s)); pB=np.zeros(len(s))
        for _,vi in GroupKFold(5,shuffle=True,random_state=seed).split(pools,groups=pools):
            vp=pools[vi]; trm=(~s.pool.isin(vp)).to_numpy(); va=~trm
            XA=[s.loc[trm&incA,FS]]; YA=[s.loc[trm&incA,'isA'].astype(int)]
            XB=[s.loc[trm&incB,FS]]; YB=[s.loc[trm&incB,'isB'].astype(int)]
            if augment:
                for r,ia,ib in REPS:
                    m=(~r.pool.isin(vp)).to_numpy()
                    XA.append(r.loc[m&ia,FS]); YA.append(r.loc[m&ia,'isA'].astype(int))
                    XB.append(r.loc[m&ib,FS]); YB.append(r.loc[m&ib,'isB'].astype(int))
            mA=lgb.LGBMClassifier(**{**C.PARAMS,'n_jobs':NJ,'random_state':seed}).fit(pd.concat(XA).astype(float),pd.concat(YA))
            mB=lgb.LGBMClassifier(**{**C.PARAMS,'n_jobs':NJ,'random_state':seed}).fit(pd.concat(XB).astype(float),pd.concat(YB))
            pA[va]=mA.predict_proba(s.loc[va,FS].astype(float))[:,1]; pB[va]=mB.predict_proba(s.loc[va,FS].astype(float))[:,1]
        L,_,_=decode(s,np.clip(pA,1e-9,1-1e-6),np.clip(pB*GB,1e-9,1-1e-6))
        aps.append(C.pair_ap(c,sc(to_c(L)),counts).reindex(idx))
    return aps
a0=run(False); a1=run(True)
d=np.mean([a1[i].values-a0[i].values for i in range(3)],axis=0)
pm=pd.Series(d,index=pool).groupby(level=0).mean(); bs=np.array([pm.values[rng.integers(0,len(pm),len(pm))].mean() for _ in range(4000)])
print(f'  baseline (full frame only): {np.mean([a.mean() for a in a0]):.4f}  seeds ' + ' '.join(f'{a.mean():.4f}' for a in a0))
print(f'  + {NREP} exposure replicas   : {np.mean([a.mean() for a in a1]):.4f}  seeds ' + ' '.join(f'{a.mean():.4f}' for a in a1))
print(f'  delta {d.mean():+.4f}  poolP={float((bs>0).mean()):.3f}',flush=True)
