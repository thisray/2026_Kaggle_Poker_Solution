"""R5-F5: pick the fourth-family hybrid threshold by SLATE QUALITY, not by the
truth's composition. R4 (and R5-f1) chose tau=0.3 from LB consistency, which
estimates how many A events the TRUTH has. But our transferred A detector is only
~50% accurate, so the optimal slate need not match the truth's composition: a 4th
marginal A pick displaces a good B pick.
Leave-one-family-out analogue of the exact hybrid construction:
   [transferred A with L_A >= tau, by L_A] + [the target family's OWN B-rule picks,
    by time] + [everything else by L]
which is exactly r29's slate with the M3-ACT picks standing in for the target's own
B rule. Sweep tau. Both directions, all co-seated hands, official AP@5."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, os, json
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18'); sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
import ci_censored_event as C
from g4_decode import decode, runs
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; NJ=int(os.environ.get('NJ',6))
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
# target-family B rules, the analogue of the M3-ACT picks used as the fourth family's own fill
def brule(d,fam):
    if fam=='directed_transfer':  # sender pays with almost no equity (big pot)
        return (((d.k_ps_eq_last<=0.3)&(d.x_conS>=5)&(d.k_pr_won>0))|((d.k_pr_eq_last<=0.3)&(d.x_conR>=5)&(d.k_ps_won>0))).values
    return ((d.sd_any.astype(bool))&(d.both_flop.astype(bool))&(d.x_s_aggr_post==0)&(d.x_r_aggr_post==0)).values  # SP: passive showdown
res={}
for tgt,src in (('directed_transfer',['soft_play']),('soft_play',['directed_transfer'])):
    key=f'{tgt[:2]}<-{src[0][:2]}'; m=(sA.fam==tgt).values; tA=sA[m].reset_index(drop=True); tF=sF[m].reset_index(drop=True)
    ss,iA,iB=types(sA[sA.fam.isin(src)].reset_index(drop=True))
    pa=np.zeros(len(tA)); pb=np.zeros(len(tA)); SD=[27,7,99,123,2026,31]
    for k,sd in enumerate(SD):
        pr={**C.PARAMS,'n_jobs':NJ,'random_state':sd}
        if k: pr.update(colsample_bytree=float(np.random.default_rng(sd).uniform(.5,.9)),subsample=.7,subsample_freq=1,num_leaves=int(np.random.default_rng(sd).choice([15,31,63])))
        pa+=both(lgb.LGBMClassifier(**pr).fit(ss.loc[iA,FS].astype(float),ss.loc[iA,'isA']),tA,tF)/len(SD)
        pb+=both(lgb.LGBMClassifier(**pr).fit(ss.loc[iB,FS].astype(float),ss.loc[iB,'isB']),tA,tF)/len(SD)
    pa=np.clip(pa,1e-6,1-1e-6); pb=np.clip(pb,1e-6,1-1e-6); L,LA,LB=decode(tA,pa,pb)
    BR=brule(tA,tgt).astype(float)
    trel=tA.groupby('slot').ts.rank(pct=True).values
    fill=BR*(2.0-trel)      # own-family B rule, earliest first (mimics M3-ACT)
    out={'trueA':round(float(tA.groupby('slot').isA.sum().mean()) if 'isA' in tA else float('nan'),2)} if False else {}
    st,_,_=types(tA); out['true_nA']=round(float(st.groupby('slot').isA.sum().mean()),2); out['true_nB']=round(float(st.groupby('slot').isB.sum().mean()),2)
    out['plainL']=round(ap(tA,L),4); out['fill_only']=round(ap(tA,fill),4)
    for tau in (0.2,0.3,0.4,0.5,0.6,0.7,0.85,1.0,1.5):
        sel=LA>=tau
        # [A by L_A] > [own-family B rule, earliest first] > [rest by L]
        k=np.where(sel, 3e6+LA, np.where(BR>0, 1e6+fill, L))
        out[f'hybrid_tau{tau}']=round(ap(tA,k),4); out[f'nA_tau{tau}']=round(float(pd.Series(sel.astype(int)).groupby(tA.slot.values).sum().mean()),2)
    res[key]=out; print(key,json.dumps(out),flush=True)
avg={k:round(float(np.mean([res[a][k] for a in res])),4) for k in res['di<-so']}
print('AVERAGE',json.dumps(avg))
json.dump({**res,'AVERAGE':avg},open(f'{O}/r5/f5_tau_lofo.json','w'),indent=1)
