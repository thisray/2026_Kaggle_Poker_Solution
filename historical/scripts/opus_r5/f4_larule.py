"""R5-F4: the fourth-family hybrid selects its front slots by L_A >= tau from the
transferred model. Does adding the family-agnostic RULE prior to L_A (not to L, as
R4's lambda test did) improve an A-priority slate on an unseen family? Measured by
leave-one-family-out on dev over all co-seated hands, i.e. the same protocol that
stands in for the fourth family."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, os, json
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18'); sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
import ci_censored_event as C
from g4_decode import decode, runs
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; NJ=int(os.environ.get('NJ',4))
def assemble(frame,role,ker):
    ROLE=[c for c in role.columns if c.startswith('x_') and c not in ('x_k','x_n','x_flow_margin')]; KER=[c for c in ker.columns if c.startswith('k_')]
    d=frame.merge(role[['slot','h']+ROLE],on=['slot','h'],validate='one_to_one').merge(ker[['slot','h']+KER],on=['slot','h'],validate='one_to_one').sort_values(['slot','ts','h'],kind='stable').reset_index(drop=True); return d,C.FEATURES+ROLE+KER
base=C.prepare(pd.read_parquet(f'{O}/t5_dev_seq.parquet'))
sA,FS=assemble(base,pd.read_parquet(f'{O}/r4/x2c_role_dev.parquet'),pd.read_parquet(f'{O}/r4/x11_kernel_dev.parquet'))
sF,_=assemble(base,pd.read_parquet(f'{O}/r4/x2cflip_role_dev.parquet'),pd.read_parquet(f'{O}/r4/x11_kernel_devflip.parquet'))
g1=pd.read_parquet(f'{O}/r4/g1_evidence_rank.parquet')[['h','pair_id','evidence_rank','chron']].sort_values(['pair_id','evidence_rank'])
g1['run']=np.concatenate([runs(g.chron.values) for _,g in g1.groupby('pair_id',sort=False)]); g1['nruns']=g1.groupby('pair_id').run.transform('max')+1
sA=sA.merge(g1[['h','run','nruns']],on='h',how='left').sort_values(['slot','ts','h'],kind='stable').reset_index(drop=True)
def types(s):
    e=s[s.ev.astype(bool)]; two=e[e.nruns==2]
    clf=lgb.LGBMClassifier(n_estimators=200,learning_rate=.05,num_leaves=7,min_child_samples=10,colsample_bytree=.5,verbosity=-1,n_jobs=NJ).fit(two[FS].astype(float),(two.run==1).astype(int))
    typB=np.where(s.nruns.values==2,s.run.values==1,clf.predict_proba(s[FS].astype(float))[:,1]>0.5)
    s=s.copy(); s['isA']=s.ev.astype(bool)&~typB; s['isB']=s.ev.astype(bool)&typB
    nA=s.groupby('slot').isA.transform('sum'); nB=s.groupby('slot').isB.transform('sum'); nev=s.groupby('slot').ev.transform('sum')
    lastA=s.slot.map(s[s.isA].groupby('slot').ts.max()); lastB=s.slot.map(s[s.isB].groupby('slot').ts.max())
    return s,((nA<5)|(s.ts<=lastA)).values,((nev<5)|((nB>0)&(s.ts<=lastB))).values
def ap_slate(t,order_key):
    """order_key: array giving the per-hand ordering score (descending)."""
    z=t[['slot','h','ts','ev']].copy(); z['s']=order_key; z=z.sort_values(['slot','s','ts'],ascending=[True,False,True]); z['r']=z.groupby('slot').cumcount()+1; z=z[z.r<=5]
    z['v']=z.ev.astype(float)*z.groupby('slot').ev.cumsum()/z.r; cnt=t.groupby('slot').ev.sum(); return float((z.groupby('slot').v.sum().reindex(cnt.index,fill_value=0)/cnt.clip(upper=5)).mean())
both=lambda m,a,f: 1-(1-m.predict_proba(a[FS].astype(float))[:,1])*(1-m.predict_proba(f[FS].astype(float))[:,1])
res={}
for tgt,src in (('directed_transfer',['soft_play']),('soft_play',['directed_transfer']),('directed_transfer',['soft_play','coordinated_isolation'])):
    key=f'{tgt[:2]}<-{"+".join(x[:2] for x in src)}'
    m=(sA.fam==tgt).values; tA=sA[m].reset_index(drop=True); tF=sF[m].reset_index(drop=True)
    ss,iA,iB=types(sA[sA.fam.isin(src)].reset_index(drop=True))
    pa=np.zeros(len(tA)); pb=np.zeros(len(tA)); SD=[27,7,99,123,2026,31]
    for k,sd in enumerate(SD):
        pr={**C.PARAMS,'n_jobs':NJ,'random_state':sd}
        if k: pr.update(colsample_bytree=float(np.random.default_rng(sd).uniform(.5,.9)),subsample=.7,subsample_freq=1,num_leaves=int(np.random.default_rng(sd).choice([15,31,63])))
        pa+=both(lgb.LGBMClassifier(**pr).fit(ss.loc[iA,FS].astype(float),ss.loc[iA,'isA']),tA,tF)/len(SD)
        pb+=both(lgb.LGBMClassifier(**pr).fit(ss.loc[iB,FS].astype(float),ss.loc[iB,'isB']),tA,tF)/len(SD)
    pa=np.clip(pa,1e-6,1-1e-6); pb=np.clip(pb,1e-6,1-1e-6)
    L,LA,LB=decode(tA,pa,pb)
    sf=(tA.x_s_fold_to_r==1)&(tA.k_pr_won>0); rf=(tA.x_r_fold_to_s==1)&(tA.k_ps_won>0)
    A2=((sf&(tA.k_ps_eq_last>=0.5))|(rf&(tA.k_pr_eq_last>=0.5))).values.astype(float)
    Bp=(((tA.k_ps_eq_last<=0.3)&(tA.x_conS>=5)&(tA.k_pr_won>0)&(tA.x_s_fold_to_r==0))|((tA.k_pr_eq_last<=0.3)&(tA.x_conR>=5)&(tA.k_ps_won>0)&(tA.x_r_fold_to_s==0))).values.astype(float)
    Lr,LAr,LBr=decode(tA,0.6*A2,0.45*Bp*(1-A2))
    out={}
    for lam in (0.0,0.25,0.5,0.75,1.0,1.5):
        LAc=LA+lam*LAr; Lc=L+lam*Lr
        out[f'plainL_lam{lam}']=round(ap_slate(tA,Lc),4)
        for tau in (0.2,0.3,0.4,0.5):
            # A-priority slate: hands with LAc>=tau first (by LAc), rest by Lc
            key2=np.where(LAc>=tau, 1e6+LAc, Lc); out[f'Apri_tau{tau}_lam{lam}']=round(ap_slate(tA,key2),4)
            if lam==0.0: out[f'nA_tau{tau}']=round(float(pd.Series((LAc>=tau).astype(int)).groupby(tA.slot.values).sum().mean()),2)
    res[key]=out; print(key,json.dumps(out),flush=True)
json.dump(res,open(f'{O}/r5/f4_larule.json','w'),indent=1)
avg={k:np.mean([res[a][k] for a in ('di<-so','so<-di')]) for k in res['di<-so'] if k in res['so<-di']}
print('AVERAGE over the two directions:'); print(json.dumps({k:round(float(v),4) for k,v in sorted(avg.items(),key=lambda kv:-kv[1])[:16]},indent=1))
