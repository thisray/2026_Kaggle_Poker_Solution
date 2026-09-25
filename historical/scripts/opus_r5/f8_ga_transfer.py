"""R5-F8: the DT win came from the decoder's sensitivity to the ABSOLUTE scale of p_B.
The fourth-family hybrid's front block uses L_A = p_A * P(#A before <= 4), which is
scale-sensitive in p_A the same way - and the transferred p_A has no reason to be
calibrated on an unseen family. Leave-one-family-out, exact hybrid construction, sweep
the p_A scale while HOLDING THE NUMBER OF A SLOTS FIXED (tau re-derived per scale) so
the scale effect is separated from the count effect."""
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
def ap(t,key):
    z=t[['slot','h','ts','ev']].copy(); z['s']=key; z=z.sort_values(['slot','s','ts'],ascending=[True,False,True]); z['r']=z.groupby('slot').cumcount()+1; z=z[z.r<=5]
    z['v']=z.ev.astype(float)*z.groupby('slot').ev.cumsum()/z.r; cnt=t.groupby('slot').ev.sum(); return float((z.groupby('slot').v.sum().reindex(cnt.index,fill_value=0)/cnt.clip(upper=5)).mean())
both=lambda m,a,f: 1-(1-m.predict_proba(a[FS].astype(float))[:,1])*(1-m.predict_proba(f[FS].astype(float))[:,1])
def brule(d,fam):
    if fam=='directed_transfer': return (((d.k_ps_eq_last<=0.3)&(d.x_conS>=5)&(d.k_pr_won>0))|((d.k_pr_eq_last<=0.3)&(d.x_conR>=5)&(d.k_ps_won>0))).values
    return ((d.sd_any.astype(bool))&(d.both_flop.astype(bool))&(d.x_s_aggr_post==0)&(d.x_r_aggr_post==0)).values
res={}
for tgt,src in (('directed_transfer',['soft_play']),('soft_play',['directed_transfer'])):
    key=f'{tgt[:2]}<-{src[0][:2]}'; m=(sA.fam==tgt).values; tA=sA[m].reset_index(drop=True); tF=sF[m].reset_index(drop=True)
    ss,iA,iB=types(sA[sA.fam.isin(src)].reset_index(drop=True))
    pa=np.zeros(len(tA)); pb=np.zeros(len(tA))
    for sd in (27,7,99):
        pa+=both(lgb.LGBMClassifier(**{**C.PARAMS,'n_jobs':NJ,'random_state':sd}).fit(ss.loc[iA,FS].astype(float),ss.loc[iA,'isA']),tA,tF)/3
        pb+=both(lgb.LGBMClassifier(**{**C.PARAMS,'n_jobs':NJ,'random_state':sd}).fit(ss.loc[iB,FS].astype(float),ss.loc[iB,'isB']),tA,tF)/3
    pa=np.clip(pa,1e-6,1-1e-6); pb=np.clip(pb,1e-6,1-1e-6)
    BR=brule(tA,tgt).astype(float); trel=tA.groupby('slot').ts.rank(pct=True).values; fill=BR*(2.0-trel)
    L0,LA0,_=decode(tA,pa,pb); TARGET=float(pd.Series((LA0>=0.3).astype(int)).groupby(tA.slot.values).sum().mean())
    out={'target_A_slots':round(TARGET,2)}
    for ga in (0.6,0.8,1.0,1.3,1.7,2.2,3.0):
        for gb in (1.0,1.5):
            L,LA,_=decode(tA,np.clip(pa*ga,1e-9,1-1e-6),np.clip(pb*gb,1e-9,1-1e-6))
            # hold the A-slot count fixed: choose tau as the quantile matching TARGET slots per pair
            thr=np.quantile(LA, 1-TARGET*tA.slot.nunique()/len(LA))
            out[f'ga{ga}_gb{gb}']=round(ap(tA,np.where(LA>=thr,3e6+LA,np.where(BR>0,1e6+fill,L))),4)
    res[key]=out; print(key,json.dumps(out),flush=True)
avg={k:round(float(np.mean([res[a][k] for a in res])),4) for k in res['di<-so']}
print('AVERAGE',json.dumps(avg)); json.dump({**res,'AVERAGE':avg},open(f'{O}/r5/f8_ga_transfer.json','w'),indent=1)
