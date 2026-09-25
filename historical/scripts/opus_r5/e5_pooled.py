"""R5-E5: the per-family learned combiner failed for one reason worth removing -
92 to 148 labelled pairs. Pool ALL THREE families (372 pairs, 7,440 candidate rows)
and give the model the family as a feature plus the raw hand features, so a
cell-conditional correction can be learned once instead of three times. Two
objectives: binary logloss and LambdaRank grouped by pair (AP@5 is a within-pair
ranking problem, which pointwise logloss does not target).
Pool GroupKFold x 3 seeds; scored with the official AP@5 per family."""
import numpy as np, pandas as pd, sys, os, json, glob, lightgbm as lgb
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18')
import ci_censored_event as C
from sklearn.model_selection import GroupKFold
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; R5=f'{O}/r5'
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9)))
FAMS={'directed_transfer':'di','soft_play':'so','coordinated_isolation':'co'}
cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
hmap=pd.read_parquet(f'{O}/np/hand_index.parquet').set_index('hand_id').hi
t45=pd.read_parquet(f'{O}/r3/t45_known_e_rerank.parquet'); t45['h']=t45.hand_id.map(hmap)
T45=[c for c in t45.columns if c in ('tab','rs_blend','sur_c_max','sur_c_mean','sur_c_aggr_max','sur_1_max','sur_1_aggr_max','nd','nag','gap_aggr','gap_max','p_base_feats_only','p_with_memo_feats','b','pa','pb')]
role=pd.read_parquet(f'{O}/r4/x2c_role_dev.parquet'); ROLE=[c for c in role.columns if c.startswith('x_') and c not in ('x_k','x_n')]
ker=pd.read_parquet(f'{O}/r4/x11_kernel_dev.parquet'); KER=[c for c in ker.columns if c.startswith('k_')]
nf=pd.read_parquet(f'{O}/r3/t58_seq_feats.parquet'); NEW=[c for c in nf.columns if c not in ('slot','h','pa','pb')]
full=C.prepare(pd.read_parquet(f'{O}/t5_dev_seq.parquet')).merge(nf[['slot','h']+NEW],on=['slot','h'],how='left')
full[NEW]=full[NEW].fillna(0.0)
full=full.merge(role[['slot','h']+ROLE],on=['slot','h']).merge(ker[['slot','h']+KER],on=['slot','h'])
blocks=[]; names=None
for fam,ab in FAMS.items():
    s=pd.read_parquet(f'{R5}/L/rows_{ab}.parquet').reset_index(drop=True)
    files=sorted(glob.glob(f'{R5}/L/{ab}__*.npy')); nm=[os.path.basename(f).split('__')[1][:-4] for f in files]
    names=nm if names is None else names; assert nm==names
    Z={n:np.load(f) for n,f in zip(nm,files)}; NS=Z[nm[0]].shape[0]
    c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev','pool']],on=['slot','h'],validate='one_to_one').merge(t45[['slot','h']+T45],on=['slot','h'],how='left').reset_index(drop=True)
    fmi=pd.MultiIndex.from_arrays([s.slot,s.h]); mi=pd.MultiIndex.from_arrays([c.slot,c.h])
    to_c=lambda v: pd.Series(v,index=fmi).reindex(mi).values
    for n in names: c[f'L_{n}']=lg(to_c(np.mean([Z[n][si][2] for si in range(NS)],axis=0)))
    c['pA']=lg(to_c(np.mean([Z[n][si][0] for n in names for si in range(NS)],axis=0)))
    c['pB']=lg(to_c(np.mean([Z[n][si][1] for n in names for si in range(NS)],axis=0)))
    c['Lmean']=np.mean([c[f'L_{n}'] for n in names],axis=0); c['Lstd']=np.std([c[f'L_{n}'] for n in names],axis=0)
    c=c.merge(full[['slot','h']+ROLE+KER+NEW+C.FEATURES].drop(columns=[x for x in C.FEATURES if x in ROLE+KER+NEW],errors='ignore'),on=['slot','h'],how='left',suffixes=('','_f'))
    c['fam']=fam; c['r15']=-c.r.values; c['ts_rel']=c.groupby('slot').ts.rank(pct=True).values
    for n in names+['Lmean','pA','pB','r15','tab']: c[f'z_{n}']=c.groupby('slot')[f'L_{n}' if n in names else n].rank(pct=True).values
    blocks.append(c)
D=pd.concat(blocks,ignore_index=True)
FEAT=[f'L_{n}' for n in names]+['pA','pB','Lmean','Lstd','r15','ts_rel']+T45+[f'z_{n}' for n in names+['Lmean','pA','pB','r15','tab']]
FEAT+=[c for c in ROLE+KER+NEW if c in D.columns]
FEAT=[c for c in dict.fromkeys(FEAT) if c in D.columns and D[c].dtype!=object]
D['famcode']=D.fam.map({f:i for i,f in enumerate(FAMS)}); FEAT2=FEAT+['famcode']
print(f'pooled rows {len(D)} pairs {D.slot.nunique()} features {len(FEAT2)}',flush=True)
pools=np.array(sorted(D.pool.unique())); y=D.ev.astype(int).values
def run(obj,feats,tag):
    oof=np.zeros(len(D))
    for seed in (11,29,260919):
        o=np.zeros(len(D))
        for _,vi in GroupKFold(5,shuffle=True,random_state=seed).split(pools,groups=pools):
            vp=pools[vi]; tr=(~D.pool.isin(vp)).to_numpy(); va=~tr
            if obj=='rank':
                dtr=D[tr].sort_values('slot',kind='stable'); g=dtr.groupby('slot',sort=False).size().values
                m=lgb.LGBMRanker(objective='lambdarank',n_estimators=400,learning_rate=.04,num_leaves=15,min_child_samples=40,colsample_bytree=.6,subsample=.8,subsample_freq=1,reg_lambda=10,label_gain=[0,1],verbosity=-1,n_jobs=8,random_state=seed)
                m.fit(dtr[feats],dtr.ev.astype(int),group=g); o[va]=m.predict(D.loc[va,feats])
            else:
                m=lgb.LGBMClassifier(n_estimators=400,learning_rate=.04,num_leaves=15,min_child_samples=40,colsample_bytree=.6,subsample=.8,subsample_freq=1,reg_lambda=10,verbosity=-1,n_jobs=8,random_state=seed).fit(D.loc[tr,feats],y[tr])
                o[va]=m.predict_proba(D.loc[va,feats])[:,1]
        oof+= (pd.Series(o).groupby(D.slot.values).rank(pct=True).values)/3
    D[f'oof_{tag}']=oof
    out={}
    for fam in FAMS:
        m=(D.fam==fam).values; c=D[m]; counts=c.groupby('slot').ev.sum()
        out[fam]=round(float(C.pair_ap(c.reset_index(drop=True),oof[m],counts).mean()),4)
    print(f'{tag:14s} {out}',flush=True); return out
res={}
for fam in FAMS:
    m=(D.fam==fam).values; c=D[m].reset_index(drop=True); counts=c.groupby('slot').ev.sum()
    res.setdefault('R15',{})[fam]=round(float(C.pair_ap(c,c.r15.values,counts).mean()),4)
    res.setdefault('Lmean_alone',{})[fam]=round(float(C.pair_ap(c,c.Lmean.values,counts).mean()),4)
print('R15        ',res['R15']); print('Lmean_alone',res['Lmean_alone'])
res['pooled_binary']=run('bin',FEAT2,'pooled_binary')
res['pooled_rank']=run('rank',FEAT2,'pooled_rank')
res['pooled_binary_small']=run('bin',[f'L_{n}' for n in names]+['pA','pB','Lmean','Lstd','r15','ts_rel','tab','famcode'],'pooled_binary_small')
res['pooled_rank_small']=run('rank',[f'L_{n}' for n in names]+['pA','pB','Lmean','Lstd','r15','ts_rel','tab','famcode'],'pooled_rank_small')
# blends of the pooled ranker with the deployed per-family reference
for tag in ('pooled_binary','pooled_rank','pooled_rank_small','pooled_binary_small'):
    for w in (0.3,0.5,0.7):
        out={}
        for fam in FAMS:
            m=(D.fam==fam).values; c=D[m].reset_index(drop=True); counts=c.groupby('slot').ev.sum()
            base=c.groupby('slot').Lmean.rank(pct=True).values if fam!='soft_play' else c.groupby('slot').r15.rank(pct=True).values
            sc=w*c[f'oof_{tag}'].values+(1-w)*base
            out[fam]=round(float(C.pair_ap(c,sc,counts).mean()),4)
        res[f'{tag}_blend{w}']=out; print(f'{tag}_blend{w}  {out}',flush=True)
D[['slot','h','fam']+[c for c in D.columns if c.startswith('oof_')]].to_parquet(f'{R5}/e5_pooled_oof.parquet')
json.dump(res,open(f'{R5}/e5_pooled.json','w'),indent=1)
