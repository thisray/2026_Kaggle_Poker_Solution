"""R5-E1b: expand the event-model zoo along two axes it never had.
 (1) FLIPPED ORIENTATION VIEW - every role/kernel feature has a sender/receiver
     orientation; the zoo only ever used x2c_role_dev. x2cflip_role_dev +
     x11_kernel_devflip is the other orientation, a genuinely different view.
 (2) CROSS-FAMILY TYPE CLASSIFIER - R5-k2 showed typing DT with a classifier
     trained on DT+SP two-run pairs is worth +0.0030 (pool P=0.891); type A is the
     same event in both families, so this doubles the typing data.
plus more hyper-parameter variety and XGBoost.
usage: FAM=... CFG=... [VIEW=flip] [XFAM=1] python e1b_zoo2.py"""
import numpy as np, pandas as pd, lightgbm as lgb, json, sys, os, time
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18'); sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
from sklearn.model_selection import GroupKFold
import ci_censored_event as C
from g4_decode import decode, runs
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; R5=f'{O}/r5'; os.makedirs(f'{R5}/L',exist_ok=True)
NJ=int(os.environ.get('NJ',4)); FAM=os.environ['FAM']; CFG=os.environ['CFG']; SEEDS=(260919,11,29)
FLIP=os.environ.get('VIEW')=='flip'; XFAM=os.environ.get('XFAM')=='1'
def assemble(rolef,kerf):
    base=C.prepare(pd.read_parquet(f'{O}/t5_dev_seq.parquet'))
    nf=pd.read_parquet(f'{O}/r3/t58_seq_feats.parquet'); NEW=[c for c in nf.columns if c not in ('slot','h','pa','pb')]
    base=base.merge(nf[['slot','h']+NEW],on=['slot','h'],how='left'); base[NEW]=base[NEW].fillna(0.0)
    role=pd.read_parquet(rolef); ROLE=[c for c in role.columns if c.startswith('x_') and c not in ('x_k','x_n')]
    ker=pd.read_parquet(kerf); KER=[c for c in ker.columns if c.startswith('k_')]
    d=base.merge(role[['slot','h']+ROLE],on=['slot','h'],validate='one_to_one').merge(ker[['slot','h']+KER],on=['slot','h'],validate='one_to_one')
    return d,C.FEATURES+NEW+ROLE+KER
RF,KF=(f'{O}/r4/x2cflip_role_dev.parquet',f'{O}/r4/x11_kernel_devflip.parquet') if FLIP else (f'{O}/r4/x2c_role_dev.parquet',f'{O}/r4/x11_kernel_dev.parquet')
full,FS0=assemble(RF,KF)
def symmetrise(d):
    out=[];cols=set(d.columns);new={}
    pairs=[(c,c.replace('k_rs_','k_sr_')) for c in cols if c.startswith('k_rs_')]+[(c,c.replace('k_pr_','k_ps_')) for c in cols if c.startswith('k_pr_')]
    pairs+=[('x_netR','x_netS'),('x_conR','x_conS'),('x_r_aggr_pre','x_s_aggr_pre'),('x_r_aggr_post','x_s_aggr_post'),('x_r_last','x_s_last'),('x_r_last_st','x_s_last_st'),('x_sdR','x_sdS'),('x_hsR_pre','x_hsS_pre'),('x_hsR_last','x_hsS_last'),('x_r_fold_to_s','x_s_fold_to_r'),('x_r_aggr_n','x_s_aggr_n')]
    for a,b in pairs:
        if a in cols and b in cols:
            nm=a.replace('k_rs_','y_i_').replace('k_pr_','y_p_').replace('x_','y_x_'); new[nm+'_mx']=np.maximum(d[a],d[b]); new[nm+'_mn']=np.minimum(d[a],d[b]); out+=[nm+'_mx',nm+'_mn']
    sf=d.x_s_fold_to_r==1; rf=d.x_r_fold_to_s==1
    new['y_folder_eq']=np.where(sf,d.k_ps_eq_last,np.where(rf,d.k_pr_eq_last,-1.0)); new['y_bettor_eq']=np.where(sf,d.k_pr_eq_last,np.where(rf,d.k_ps_eq_last,-1.0))
    new['y_folder_hs']=np.where(sf,d.x_hsS_last,np.where(rf,d.x_hsR_last,-1.0)); new['y_folder_contrib']=np.where(sf,d.x_conS,np.where(rf,d.x_conR,-1.0)); new['y_eq_gap_abs']=(d.k_ps_eq_last-d.k_pr_eq_last).abs()
    return pd.concat([d,pd.DataFrame(new,index=d.index)],axis=1),out+['y_folder_eq','y_bettor_eq','y_folder_hs','y_folder_contrib','y_eq_gap_abs']
full,SY=symmetrise(full); FS=FS0+SY
g1=pd.read_parquet(f'{O}/r4/g1_evidence_rank.parquet')[['h','pair_id','evidence_rank','chron']].sort_values(['pair_id','evidence_rank'])
g1['run']=np.concatenate([runs(g.chron.values) for _,g in g1.groupby('pair_id',sort=False)]); g1['nruns']=g1.groupby('pair_id').run.transform('max')+1
full=full.merge(g1[['h','run','nruns']],on='h',how='left').sort_values(['slot','ts','h'],kind='stable').reset_index(drop=True)
CFGS={
 'x_base':dict(C.PARAMS),
 'x_deep':{**C.PARAMS,'num_leaves':127,'min_child_samples':15,'n_estimators':400,'learning_rate':.03},
 'x_shal':{**C.PARAMS,'num_leaves':5,'n_estimators':1200,'learning_rate':.02,'min_child_samples':80},
 'x_goss':{**C.PARAMS,'boosting_type':'goss','num_leaves':31,'n_estimators':500,'random_state':5},
 'x_extra':{**C.PARAMS,'extra_trees':True,'num_leaves':63,'n_estimators':700,'random_state':13},
 'x_col2':{**C.PARAMS,'colsample_bytree':.2,'num_leaves':31,'n_estimators':800,'random_state':17},
 'x_a1':{**C.PARAMS,'reg_alpha':5,'reg_lambda':50,'num_leaves':31,'n_estimators':600,'random_state':23},
 'x_bag5':{**C.PARAMS,'subsample':.5,'subsample_freq':1,'colsample_bytree':.5,'num_leaves':31,'n_estimators':700,'random_state':29},
 'x_mcs5':{**C.PARAMS,'min_child_samples':5,'num_leaves':31,'n_estimators':500,'random_state':31},
 'xgb':'xgboost','cat2':'catboost2'}
par=CFGS[CFG]
s=full[full.fam==FAM].reset_index(drop=True); e=s[s.ev.astype(bool)]
if FAM=='coordinated_isolation': typB=(s.pa_at_trig<6).values
else:
    tf=['directed_transfer','soft_play'] if XFAM else [FAM]
    et=full[full.fam.isin(tf)&full.ev.astype(bool)]; two=et[et.nruns==2]
    clf=lgb.LGBMClassifier(n_estimators=200,learning_rate=.05,num_leaves=7,min_child_samples=10,colsample_bytree=.5,verbosity=-1,n_jobs=NJ).fit(two[FS].astype(float),(two.run==1).astype(int))
    typB=np.where(s.nruns.values==2,s.run.values==1,clf.predict_proba(s[FS].astype(float))[:,1]>0.5)
s['isA']=s.ev.astype(bool)&~typB; s['isB']=s.ev.astype(bool)&typB
nA=s.groupby('slot').isA.transform('sum'); nB=s.groupby('slot').isB.transform('sum'); nev=s.groupby('slot').ev.transform('sum')
lastA=s.slot.map(s[s.isA].groupby('slot').ts.max()); lastB=s.slot.map(s[s.isB].groupby('slot').ts.max())
incA=((nA<5)|(s.ts<=lastA)).values; incB=((nev<5)|((nB>0)&(s.ts<=lastB))).values
pools=np.array(sorted(full.pool.unique()))
def mk(p,seed):
    if p=='xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=500,max_depth=6,learning_rate=.05,subsample=.8,colsample_bytree=.6,reg_lambda=10,n_jobs=NJ,random_state=seed,tree_method='hist',eval_metric='logloss')
    if p=='catboost2':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=800,depth=8,learning_rate=.04,l2_leaf_reg=10,verbose=0,thread_count=NJ,random_seed=seed)
    return lgb.LGBMClassifier(**{**p,'n_jobs':NJ,'random_state':seed})
t0=time.time(); keep=[]
for seed in SEEDS:
    pA=np.zeros(len(s)); pB=np.zeros(len(s))
    for _,vi in GroupKFold(5,shuffle=True,random_state=seed).split(pools,groups=pools):
        vp=pools[vi]; tr=(~s.pool.isin(vp)).to_numpy(); va=~tr
        pA[va]=mk(par,seed).fit(s.loc[tr&incA,FS].astype(float),s.loc[tr&incA,'isA'].astype(int)).predict_proba(s.loc[va,FS].astype(float))[:,1]
        if s.loc[tr&incB,'isB'].sum()>=20:
            pB[va]=mk(par,seed).fit(s.loc[tr&incB,FS].astype(float),s.loc[tr&incB,'isB'].astype(int)).predict_proba(s.loc[va,FS].astype(float))[:,1]
    keep.append(np.stack([pA,pB,decode(s,np.clip(pA,0,1-1e-6),np.clip(pB,0,1-1e-6))[0]]))
tag=CFG+('_flip' if FLIP else '')+('_xf' if XFAM else '')
np.save(f'{R5}/L/{FAM[:2]}__{tag}.npy',np.stack(keep)); print(f'{FAM} {tag} done in {time.time()-t0:.0f}s',flush=True)
