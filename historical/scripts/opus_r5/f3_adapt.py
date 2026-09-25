"""R5-F3: can leave-one-family-out transfer be improved? The fourth family has no
labels, so LOFO on dev (train on one A-type family, score the other over ALL
co-seated hands, both orientations) is the only local proxy for the fourth-family
slate quality. R4 measured the baseline (SP->DT 0.50, DT->SP 0.47) but tried no
adaptation. Variants:
  base    source model only (R4)
  bag     bagged source models (seeds x subsample x colsample) - less source-specific
  rule    + lambda * rule decoder (R4 found 0.5 helps SP->DT)
  self    pseudo-label self-training ON THE TARGET, pool-OOF so a pool never sees
          its own pseudo-labels; soft weights from the listing posterior
  self+   self-training blended back with the source model
Everything is evaluated with the official AP@5 over all co-seated hands."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, os, json
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18'); sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
import ci_censored_event as C
from g4_decode import decode, runs
from sklearn.model_selection import GroupKFold
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
def ap_all(t,score):
    z=t[['slot','h','ts','ev']].copy(); z['s']=score; z=z.sort_values(['slot','s','ts'],ascending=[True,False,True]); z['r']=z.groupby('slot').cumcount()+1; z=z[z.r<=5]
    z['v']=z.ev.astype(float)*z.groupby('slot').ev.cumsum()/z.r; cnt=t.groupby('slot').ev.sum(); return float((z.groupby('slot').v.sum().reindex(cnt.index,fill_value=0)/cnt.clip(upper=5)).mean())
def both(m,tA,tF):
    return 1-(1-m.predict_proba(tA[FS].astype(float))[:,1])*(1-m.predict_proba(tF[FS].astype(float))[:,1])
res={}
for tgt,src in (('directed_transfer',['soft_play']),('soft_play',['directed_transfer'])):
    m=(sA.fam==tgt).values; tA=sA[m].reset_index(drop=True); tF=sF[m].reset_index(drop=True)
    ss,iA,iB=types(sA[sA.fam.isin(src)].reset_index(drop=True))
    out={}
    # ---- base + bag ----
    P={}
    for nm,seeds in (('base',[27]),('bag',[27,7,99,123,2026,31])):
        pa=np.zeros(len(tA)); pb=np.zeros(len(tA))
        for k,sd in enumerate(seeds):
            pr={**C.PARAMS,'n_jobs':NJ,'random_state':sd}
            if k: pr.update(colsample_bytree=float(np.random.default_rng(sd).uniform(.5,.9)),subsample=.7,subsample_freq=1,num_leaves=int(np.random.default_rng(sd).choice([15,31,63])))
            MA=lgb.LGBMClassifier(**pr).fit(ss.loc[iA,FS].astype(float),ss.loc[iA,'isA']); MB=lgb.LGBMClassifier(**pr).fit(ss.loc[iB,FS].astype(float),ss.loc[iB,'isB'])
            pa+=both(MA,tA,tF)/len(seeds); pb+=both(MB,tA,tF)/len(seeds)
        P[nm]=(np.clip(pa,1e-6,1-1e-6),np.clip(pb,1e-6,1-1e-6))
        L,LA,LB=decode(tA,*P[nm]); out[nm]=round(ap_all(tA,L),4); out[nm+'_Aonly']=round(ap_all(tA,LA),4)
    # ---- rule decoder blend on top of bag ----
    sf=(tA.x_s_fold_to_r==1)&(tA.k_pr_won>0); rf=(tA.x_r_fold_to_s==1)&(tA.k_ps_won>0)
    A2=((sf&(tA.k_ps_eq_last>=0.5))|(rf&(tA.k_pr_eq_last>=0.5))).values.astype(float)
    Bp=(((tA.k_ps_eq_last<=0.3)&(tA.x_conS>=5)&(tA.k_pr_won>0)&(tA.x_s_fold_to_r==0))|((tA.k_pr_eq_last<=0.3)&(tA.x_conR>=5)&(tA.k_ps_won>0)&(tA.x_r_fold_to_s==0))).values.astype(float)
    Lr,_,_=decode(tA,0.6*A2,0.45*Bp*(1-A2))
    Lbag,_,_=decode(tA,*P['bag'])
    for lam in (0.25,0.5,1.0): out[f'bag+rule{lam}']=round(ap_all(tA,Lbag+lam*Lr),4)
    # ---- pseudo-label self-training on the target, pool-OOF ----
    pools=np.array(sorted(tA.pool.unique()))
    for srcname in ('bag',):
        pa,pb=P[srcname]; L0,LA0,LB0=decode(tA,pa,pb)
        d=tA[['slot','pool']].copy(); d['L']=L0; d['LA']=LA0
        d['rk']=d.groupby('slot').L.rank(ascending=False,method='first')
        yps=(d.rk<=5).values.astype(int); wps=np.clip(L0,0.02,1.0)
        ypsA=((d.rk<=5)&(LA0>=LB0)).values.astype(int)
        for tag,yy in (('self',yps),('selfA',ypsA)):
            p1=np.zeros(len(tA))
            for _,vi in GroupKFold(5,shuffle=True,random_state=260919).split(pools,groups=pools):
                vp=pools[vi]; tr=(~tA.pool.isin(vp)).to_numpy(); va=~tr
                mm=lgb.LGBMClassifier(**{**C.PARAMS,'n_jobs':NJ}).fit(tA.loc[tr,FS].astype(float),yy[tr],sample_weight=np.where(yy[tr]==1,wps[tr],1.0))
                p1[va]=both(mm,tA[va].reset_index(drop=True),tF[va].reset_index(drop=True))
            for w in (0.0,0.3,0.5,0.7,1.0):
                pmix=np.clip((1-w)*pa+w*p1,1e-6,1-1e-6) if tag=='self' else np.clip((1-w)*pa+w*p1,1e-6,1-1e-6)
                Lx,_,_=decode(tA,pmix,pb); out[f'{tag}_w{w}']=round(ap_all(tA,Lx),4)
    res[tgt]=out; print(tgt,json.dumps(out),flush=True)
json.dump(res,open(f'{O}/r5/f3_adapt.json','w'),indent=1)
