"""R5-F2: transfer EFFICIENCY. R4 measured leave-one-family-out transfer (SP->DT 0.50,
DT->SP 0.47) but never the in-family ceiling in the SAME protocol (all co-seated hands,
no R15 shortlist, both orientations). Without that the fourth-family gain cannot be
estimated: efficiency = LOFO / self is what carries over."""
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
    if (s.fam=='coordinated_isolation').any(): typB=np.where(s.fam.values=='coordinated_isolation',s.pa_at_trig.values<6,typB)
    s=s.copy(); s['isA']=s.ev.astype(bool)&~typB; s['isB']=s.ev.astype(bool)&typB
    nA=s.groupby('slot').isA.transform('sum'); nB=s.groupby('slot').isB.transform('sum'); nev=s.groupby('slot').ev.transform('sum')
    lastA=s.slot.map(s[s.isA].groupby('slot').ts.max()); lastB=s.slot.map(s[s.isB].groupby('slot').ts.max())
    return s,((nA<5)|(s.ts<=lastA)).values,((nev<5)|((nB>0)&(s.ts<=lastB))).values
def ap_all(t,score):
    z=t[['slot','h','ts','ev']].copy(); z['s']=score; z=z.sort_values(['slot','s','ts'],ascending=[True,False,True]); z['r']=z.groupby('slot').cumcount()+1; z=z[z.r<=5]
    z['v']=z.ev.astype(float)*z.groupby('slot').ev.cumsum()/z.r; cnt=t.groupby('slot').ev.sum(); return float((z.groupby('slot').v.sum().reindex(cnt.index,fill_value=0)/cnt.clip(upper=5)).mean())
pools=np.array(sorted(sA.pool.unique())); res={}
for tgt in ('directed_transfer','soft_play','coordinated_isolation'):
    m=(sA.fam==tgt).values; tA=sA[m].reset_index(drop=True); tF=sF[m].reset_index(drop=True)
    st,incA,incB=types(sA[m].reset_index(drop=True))
    # --- self, pool-OOF, both orientations (same protocol as the transfer) ---
    pa=np.zeros(m.sum()); pb=np.zeros(m.sum())
    for _,vp_i in GroupKFold(5,shuffle=True,random_state=260919).split(pools,groups=pools):
        vp=pools[vp_i]; tr=(~st.pool.isin(vp)).to_numpy(); va=~tr
        mA=lgb.LGBMClassifier(**{**C.PARAMS,'n_jobs':NJ}).fit(st.loc[tr&incA,FS].astype(float),st.loc[tr&incA,'isA'])
        mB=lgb.LGBMClassifier(**{**C.PARAMS,'n_jobs':NJ}).fit(st.loc[tr&incB,FS].astype(float),st.loc[tr&incB,'isB'])
        for mm,acc in ((mA,'a'),(mB,'b')):
            p1=mm.predict_proba(tA.loc[va,FS].astype(float))[:,1]; p2=mm.predict_proba(tF.loc[va,FS].astype(float))[:,1]
            (pa if acc=='a' else pb)[va]=1-(1-p1)*(1-p2)
    Lself,_,_=decode(tA,np.clip(pa,0,1-1e-6),np.clip(pb,0,1-1e-6))
    # --- LOFO ---
    src=[f for f in ('directed_transfer','soft_play','coordinated_isolation') if f!=tgt]
    ss,iA,iB=types(sA[sA.fam.isin(src)].reset_index(drop=True))
    MA=lgb.LGBMClassifier(**{**C.PARAMS,'n_jobs':NJ}).fit(ss.loc[iA,FS].astype(float),ss.loc[iA,'isA'])
    MB=lgb.LGBMClassifier(**{**C.PARAMS,'n_jobs':NJ}).fit(ss.loc[iB,FS].astype(float),ss.loc[iB,'isB'])
    qa=1-(1-MA.predict_proba(tA[FS].astype(float))[:,1])*(1-MA.predict_proba(tF[FS].astype(float))[:,1])
    qb=1-(1-MB.predict_proba(tA[FS].astype(float))[:,1])*(1-MB.predict_proba(tF[FS].astype(float))[:,1])
    Llofo,_,_=decode(tA,np.clip(qa,0,1-1e-6),np.clip(qb,0,1-1e-6))
    res[tgt]=dict(self_oof=round(ap_all(tA,Lself),4), lofo=round(ap_all(tA,Llofo),4), time_only=round(ap_all(tA,-tA.ts.values),4), pairs=int(tA.slot.nunique()))
    res[tgt]['efficiency']=round(res[tgt]['lofo']/res[tgt]['self_oof'],3)
    print(tgt,res[tgt],flush=True)
json.dump(res,open(f'{O}/r5/f2_lofo_self.json','w'),indent=1)
