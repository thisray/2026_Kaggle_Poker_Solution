"""R5-E1: a zoo of diverse typed event models per family, saved as OOF listing
probabilities L (one file per config). Same protocol as R4-G4 so the numbers are
directly comparable: pool GroupKFold(5) x 3 seeds, decode with the two-type
listing decoder, official AP@5 inside the frozen R15 top-20 with full-truth
denominators.

Motivation: the P lane went from 3 to 64 models for +0.006 AP. The evidence lane
still runs on 3 (R15 ranker, TabICL, one typed decoder) while sitting 0.23 below
its own oracle, so model diversity is far less exploited there.

usage: FAM=directed_transfer CFG=lgb_base python e1_zoo.py
"""
import numpy as np, pandas as pd, lightgbm as lgb, json, sys, os, time
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18')
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
from sklearn.model_selection import GroupKFold
import ci_censored_event as C
from g4_decode import decode, runs
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'
R5=f'{O}/r5'; os.makedirs(f'{R5}/L',exist_ok=True)
NJ=int(os.environ.get('NJ',4)); FAM=os.environ['FAM']; CFG=os.environ['CFG']
SEEDS=(260919,11,29)

def assemble(role_f,ker_f):
    base=C.prepare(pd.read_parquet(f'{O}/t5_dev_seq.parquet'))
    nf=pd.read_parquet(f'{O}/r3/t58_seq_feats.parquet'); NEW=[c for c in nf.columns if c not in ('slot','h','pa','pb')]
    base=base.merge(nf[['slot','h']+NEW],on=['slot','h'],how='left'); base[NEW]=base[NEW].fillna(0.0)
    role=pd.read_parquet(role_f); ROLE=[c for c in role.columns if c.startswith('x_') and c not in ('x_k','x_n')]
    ker=pd.read_parquet(ker_f); KER=[c for c in ker.columns if c.startswith('k_')]
    d=base.merge(role[['slot','h']+ROLE],on=['slot','h'],validate='one_to_one').merge(ker[['slot','h']+KER],on=['slot','h'],validate='one_to_one')
    return d, C.FEATURES+NEW+ROLE+KER, NEW, ROLE, KER

full,FS_ALL,NEW,ROLE,KER = assemble(f'{O}/r4/x2c_role_dev.parquet', f'{O}/r4/x11_kernel_dev.parquet')
g1=pd.read_parquet(f'{O}/r4/g1_evidence_rank.parquet')[['h','pair_id','evidence_rank','chron']].sort_values(['pair_id','evidence_rank'])
g1['run']=np.concatenate([runs(g.chron.values) for _,g in g1.groupby('pair_id',sort=False)]); g1['nruns']=g1.groupby('pair_id').run.transform('max')+1
full=full.merge(g1[['h','evidence_rank','run','nruns']],on='h',how='left').sort_values(['slot','ts','h'],kind='stable').reset_index(drop=True)

def symmetrise(d):
    out=[];cols=set(d.columns)
    pairs=[(c,c.replace('k_rs_','k_sr_')) for c in cols if c.startswith('k_rs_')]+[(c,c.replace('k_pr_','k_ps_')) for c in cols if c.startswith('k_pr_')]
    pairs+=[('x_netR','x_netS'),('x_conR','x_conS'),('x_r_aggr_pre','x_s_aggr_pre'),('x_r_aggr_post','x_s_aggr_post'),('x_r_last','x_s_last'),('x_r_last_st','x_s_last_st'),('x_sdR','x_sdS'),('x_hsR_pre','x_hsS_pre'),('x_hsR_last','x_hsS_last'),('x_r_fold_to_s','x_s_fold_to_r'),('x_r_aggr_n','x_s_aggr_n')]
    for a,b in pairs:
        if a in cols and b in cols:
            nm=a.replace('k_rs_','y_i_').replace('k_pr_','y_p_').replace('x_','y_x_'); d[nm+'_mx']=np.maximum(d[a],d[b]); d[nm+'_mn']=np.minimum(d[a],d[b]); out+=[nm+'_mx',nm+'_mn']
    sf=d.x_s_fold_to_r==1; rf=d.x_r_fold_to_s==1
    d['y_folder_eq']=np.where(sf,d.k_ps_eq_last,np.where(rf,d.k_pr_eq_last,-1.0)); d['y_bettor_eq']=np.where(sf,d.k_pr_eq_last,np.where(rf,d.k_ps_eq_last,-1.0)); d['y_folder_hs']=np.where(sf,d.x_hsS_last,np.where(rf,d.x_hsR_last,-1.0))
    d['y_folder_contrib']=np.where(sf,d.x_conS,np.where(rf,d.x_conR,-1.0)); d['y_eq_gap_abs']=(d.k_ps_eq_last-d.k_pr_eq_last).abs()
    return out+['y_folder_eq','y_bettor_eq','y_folder_hs','y_folder_contrib','y_eq_gap_abs']
SYMC=symmetrise(full)

CFGS={
 'lgb_base':   dict(view='all', p=dict(C.PARAMS)),
 'lgb_sym':    dict(view='all+sym', p=dict(C.PARAMS)),
 'lgb_deep':   dict(view='all+sym', p={**C.PARAMS,'num_leaves':63,'min_child_samples':20,'n_estimators':500,'learning_rate':.03}),
 'lgb_shal':   dict(view='all+sym', p={**C.PARAMS,'num_leaves':7,'n_estimators':900,'learning_rate':.02,'min_child_samples':60}),
 'lgb_goss':   dict(view='all+sym', p={**C.PARAMS,'boosting_type':'goss','num_leaves':31,'n_estimators':500}),
 'lgb_dart':   dict(view='all+sym', p={**C.PARAMS,'boosting_type':'dart','n_estimators':400,'learning_rate':.06,'drop_rate':.1}),
 'lgb_extra':  dict(view='all+sym', p={**C.PARAMS,'extra_trees':True,'num_leaves':31,'n_estimators':600}),
 'lgb_col3':   dict(view='all+sym', p={**C.PARAMS,'colsample_bytree':.3,'num_leaves':31,'n_estimators':600,'random_state':7}),
 'lgb_kern':   dict(view='kern',p=dict(C.PARAMS)),
 'lgb_gplay':  dict(view='gplay',p=dict(C.PARAMS)),
 'lgb_role':   dict(view='role',p=dict(C.PARAMS)),
 'lgb_l2':     dict(view='all+sym', p={**C.PARAMS,'reg_lambda':200,'num_leaves':31,'n_estimators':600,'random_state':99}),
 'lgb_bag':    dict(view='all+sym', p={**C.PARAMS,'subsample':.6,'subsample_freq':1,'num_leaves':31,'n_estimators':600,'random_state':123}),
 'cat':        dict(view='all+sym', p='catboost'),
}
cfg=CFGS[CFG]
VIEWS={'all':FS_ALL,'all+sym':FS_ALL+SYMC,'kern':KER+['ts'],'gplay':C.FEATURES+NEW,'role':ROLE+NEW}
FS=[c for c in VIEWS[cfg['view']] if c in full.columns]
print(f'{FAM} {CFG}: {len(FS)} features',flush=True)

s=full[full.fam==FAM].reset_index(drop=True); e=s[s.ev.astype(bool)]
if FAM=='coordinated_isolation': typB=(s.pa_at_trig<6).values
else:
    two=e[e.nruns==2]
    clf=lgb.LGBMClassifier(n_estimators=200,learning_rate=.05,num_leaves=7,min_child_samples=10,colsample_bytree=.5,verbosity=-1,n_jobs=NJ).fit(two[FS].astype(float),(two.run==1).astype(int))
    typB=np.where(s.nruns.values==2, s.run.values==1, clf.predict_proba(s[FS].astype(float))[:,1]>0.5)
s['isA']=s.ev.astype(bool)&~typB; s['isB']=s.ev.astype(bool)&typB
nA=s.groupby('slot').isA.transform('sum'); nB=s.groupby('slot').isB.transform('sum'); nev=s.groupby('slot').ev.transform('sum')
lastA=s.slot.map(s[s.isA].groupby('slot').ts.max()); lastB=s.slot.map(s[s.isB].groupby('slot').ts.max())
incA=((nA<5)|(s.ts<=lastA)).values; incB=((nev<5)|((nB>0)&(s.ts<=lastB))).values
pools=np.array(sorted(full.pool.unique()))

def mk(p,seed):
    if p=='catboost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=600,depth=6,learning_rate=.05,l2_leaf_reg=6,verbose=0,thread_count=NJ,random_seed=seed)
    return lgb.LGBMClassifier(**{**p,'n_jobs':NJ,'random_state':seed})

t0=time.time(); keep=[]
for seed in SEEDS:
    pA=np.zeros(len(s)); pB=np.zeros(len(s))
    for _,va_pool in GroupKFold(5,shuffle=True,random_state=seed).split(pools,groups=pools):
        vp=pools[va_pool]; tr=(~s.pool.isin(vp)).to_numpy(); va=~tr
        mA=mk(cfg['p'],seed).fit(s.loc[tr&incA,FS].astype(float),s.loc[tr&incA,'isA'].astype(int)); pA[va]=mA.predict_proba(s.loc[va,FS].astype(float))[:,1]
        if s.loc[tr&incB,'isB'].sum()>=20:
            mB=mk(cfg['p'],seed).fit(s.loc[tr&incB,FS].astype(float),s.loc[tr&incB,'isB'].astype(int)); pB[va]=mB.predict_proba(s.loc[va,FS].astype(float))[:,1]
    L,LA,LB=decode(s,np.clip(pA,0,1-1e-6),np.clip(pB,0,1-1e-6)); keep.append(np.stack([pA,pB,L]))
np.save(f'{R5}/L/{FAM[:2]}__{CFG}.npy',np.stack(keep))
s[['slot','h','ts','ev','pool','isA','isB']].to_parquet(f'{R5}/L/rows_{FAM[:2]}.parquet')
print(f'{FAM} {CFG} done in {time.time()-t0:.0f}s',flush=True)
