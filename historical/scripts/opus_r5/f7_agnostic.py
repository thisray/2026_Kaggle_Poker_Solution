"""R5-F7: the fourth-family slate uses only the transferred two-type model plus the
M3-ACT picks. The known families additionally get a family-AGNOSTIC candidate ranker
(R15) and TabICL, worth about +0.08 to them. Does a family-agnostic ranker carry
signal to an UNSEEN family? Train a plain hand-level evidence ranker on the two source
families only (no target-family labels at all) and test whether blending it into the
hybrid slate helps. Leave-one-family-out, all co-seated hands, official AP@5."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, os, json
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18'); sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
import ci_censored_event as C
from g4_decode import decode, runs
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; NJ=int(os.environ.get('NJ',6))
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9)))
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
def ap(t,key):
    z=t[['slot','h','ts','ev']].copy(); z['s']=key; z=z.sort_values(['slot','s','ts'],ascending=[True,False,True]); z['r']=z.groupby('slot').cumcount()+1; z=z[z.r<=5]
    z['v']=z.ev.astype(float)*z.groupby('slot').ev.cumsum()/z.r; cnt=t.groupby('slot').ev.sum(); return float((z.groupby('slot').v.sum().reindex(cnt.index,fill_value=0)/cnt.clip(upper=5)).mean())
both=lambda m,a,f: 1-(1-m.predict_proba(a[FS].astype(float))[:,1])*(1-m.predict_proba(f[FS].astype(float))[:,1])
def brule(d,fam):
    if fam=='directed_transfer': return (((d.k_ps_eq_last<=0.3)&(d.x_conS>=5)&(d.k_pr_won>0))|((d.k_pr_eq_last<=0.3)&(d.x_conR>=5)&(d.k_ps_won>0))).values
    return ((d.sd_any.astype(bool))&(d.both_flop.astype(bool))&(d.x_s_aggr_post==0)&(d.x_r_aggr_post==0)).values
res={}
for tgt,src in (('directed_transfer',['soft_play','coordinated_isolation']),('soft_play',['directed_transfer','coordinated_isolation'])):
    key=f'{tgt[:2]}<-agn'; m=(sA.fam==tgt).values; tA=sA[m].reset_index(drop=True); tF=sF[m].reset_index(drop=True)
    # two-type transfer from the A-type source only (as the deployed patch does)
    asrc=[f for f in src if f!='coordinated_isolation']
    ss,iA,iB=types(sA[sA.fam.isin(asrc)].reset_index(drop=True))
    pa=np.zeros(len(tA)); pb=np.zeros(len(tA))
    for sd in (27,7,99):
        pa+=both(lgb.LGBMClassifier(**{**C.PARAMS,'n_jobs':NJ,'random_state':sd}).fit(ss.loc[iA,FS].astype(float),ss.loc[iA,'isA']),tA,tF)/3
        pb+=both(lgb.LGBMClassifier(**{**C.PARAMS,'n_jobs':NJ,'random_state':sd}).fit(ss.loc[iB,FS].astype(float),ss.loc[iB,'isB']),tA,tF)/3
    pa=np.clip(pa,1e-6,1-1e-6); pb=np.clip(pb,1e-6,1-1e-6); L,LA,LB=decode(tA,pa,pb)
    # family-agnostic ranker: plain 'is this hand evidence' on ALL source families (both orientations), no typing, no decoder
    gs=sA[sA.fam.isin(src)].reset_index(drop=True); gsF=sF[sA.fam.isin(src).values].reset_index(drop=True)
    Xg=pd.concat([gs[FS],gsF[FS]]); yg=pd.concat([gs.ev.astype(int),gs.ev.astype(int)])
    AG=np.zeros(len(tA))
    for sd in (27,7,99):
        AG+=both(lgb.LGBMClassifier(**{**C.PARAMS,'n_jobs':NJ,'random_state':sd}).fit(Xg.astype(float),yg),tA,tF)/3
    AGq=C.first_k_marginal(tA,np.clip(AG,0,1))     # same first-k decoder R15-style scores get
    BR=brule(tA,tgt).astype(float); trel=tA.groupby('slot').ts.rank(pct=True).values; fill=BR*(2.0-trel)
    rk=lambda v: pd.Series(v).groupby(tA.slot.values).rank(pct=True).values
    out={'agnostic_alone':round(ap(tA,AG),4),'agnostic_firstk':round(ap(tA,AGq),4),'typed_L':round(ap(tA,L),4)}
    for tau in (0.3,0.4):
        out[f'hybrid_tau{tau}']=round(ap(tA,np.where(LA>=tau,3e6+LA,np.where(BR>0,1e6+fill,L))),4)
        for w in (0.2,0.35,0.5):
            LAb=(1-w)*rk(LA)+w*rk(AGq); Lb=(1-w)*rk(L)+w*rk(AGq)
            thr=np.quantile(rk(LA),1-np.mean(LA>=tau))    # keep the same number of A slots
            out[f'hybrid_tau{tau}_agn{w}']=round(ap(tA,np.where(rk(LA)>=thr,3e6+LAb,np.where(BR>0,1e6+fill,Lb))),4)
    res[key]=out; print(key,json.dumps(out),flush=True)
avg={k:round(float(np.mean([res[a][k] for a in res])),4) for k in res[list(res)[0]]}
print('AVERAGE',json.dumps(avg)); json.dump({**res,'AVERAGE':avg},open(f'{O}/r5/f7_agnostic.json','w'),indent=1)
