"""R5-M1: the fourth-family slate keeps the two signals in SEPARATE SLOTS - transferred
type-A events first, then the partner-card (M3-ACT) picks. It never combines them per
hand. c24_hand_tables.parquet holds the same partner-card-dependence ratios for the dev
known-family pairs, so the combination can be tested leave-one-family-out: does a per-hand
blend of the transferred L_A with the partner-card ratio q beat the slot-separated hybrid?"""
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
q=pd.read_parquet(f'{O}/c24_hand_tables.parquet')
QC=[c for c in q.columns if c.startswith('q_')]+['pair_win']
print('c24 rows',len(q),'pairs',q.slot.nunique(),'q columns',QC,flush=True)
def types(s):
    e=s[s.ev.astype(bool)]; two=e[e.nruns==2]
    clf=lgb.LGBMClassifier(n_estimators=200,learning_rate=.05,num_leaves=7,min_child_samples=10,colsample_bytree=.5,verbosity=-1,n_jobs=NJ).fit(two[FS].astype(float),(two.run==1).astype(int))
    typB=np.where(s.nruns.values==2,s.run.values==1,clf.predict_proba(s[FS].astype(float))[:,1]>0.5)
    s=s.copy(); s['isA']=s.ev.astype(bool)&~typB; s['isB']=s.ev.astype(bool)&typB
    nA=s.groupby('slot').isA.transform('sum'); nB=s.groupby('slot').isB.transform('sum'); nev=s.groupby('slot').ev.transform('sum')
    lastA=s.slot.map(s[s.isA].groupby('slot').ts.max()); lastB=s.slot.map(s[s.isB].groupby('slot').ts.max())
    return s,((nA<5)|(s.ts<=lastA)).values,((nev<5)|((nB>0)&(s.ts<=lastB))).values
both=lambda m,a,f: 1-(1-m.predict_proba(a[FS].astype(float))[:,1])*(1-m.predict_proba(f[FS].astype(float))[:,1])
res={}
for tgt,src in (('directed_transfer',['soft_play']),('soft_play',['directed_transfer'])):
    key=f'{tgt[:2]}<-{src[0][:2]}'; m=(sA.fam==tgt).values; tA=sA[m].reset_index(drop=True); tF=sF[m].reset_index(drop=True)
    ss,iA,iB=types(sA[sA.fam.isin(src)].reset_index(drop=True))
    pa=np.zeros(len(tA)); pb=np.zeros(len(tA))
    for sd in (27,7,99):
        pa+=both(lgb.LGBMClassifier(**{**C.PARAMS,'n_jobs':NJ,'random_state':sd}).fit(ss.loc[iA,FS].astype(float),ss.loc[iA,'isA']),tA,tF)/3
        pb+=both(lgb.LGBMClassifier(**{**C.PARAMS,'n_jobs':NJ,'random_state':sd}).fit(ss.loc[iB,FS].astype(float),ss.loc[iB,'isB']),tA,tF)/3
    pa=np.clip(pa,1e-6,1-1e-6); pb=np.clip(pb,1e-6,1-1e-6); L,LA,LB=decode(tA,pa,pb)
    t=tA[['slot','h','ts','ev']].copy(); t['L']=L; t['LA']=LA
    j=t.merge(q[['slot','h']+QC],on=['slot','h'],how='inner')
    cnt=j.groupby('slot').ev.sum()
    print(f'{key}: c24-covered hands {len(j)} of {len(t)}; pairs {j.slot.nunique()}; truth inside {int(j.ev.sum())} of {int(t.ev.sum())}',flush=True)
    def ap(key_):
        z=j[['slot','h','ts','ev']].copy(); z['s']=key_; z=z.sort_values(['slot','s','ts'],ascending=[True,False,True]); z['r']=z.groupby('slot').cumcount()+1; z=z[z.r<=5]
        z['v']=z.ev.astype(float)*z.groupby('slot').ev.cumsum()/z.r
        return float((z.groupby('slot').v.sum().reindex(cnt.index,fill_value=0)/cnt.clip(upper=5)).mean())
    rk=lambda v: pd.Series(v).groupby(j.slot.values).rank(pct=True).values
    trel=j.groupby('slot').ts.rank(pct=True).values
    out={'L_alone':round(ap(j.L.values),4)}
    for c in QC:
        out[f'q_{c}_alone']=round(ap(j[c].values),4)
    best_q=max([c for c in QC if c!='pair_win'],key=lambda c: out[f'q_{c}_alone'])
    out['best_q']=best_q
    Q=j[best_q].values
    # (a) slot-separated hybrid: A first by L_A, then partner-card picks earliest-first, then L
    for tau in (0.3,0.4,0.5):
        sel=j.LA.values>=tau; fill=(Q>np.quantile(Q,0.8))*(2.0-trel)
        out[f'hybrid_tau{tau}']=round(ap(np.where(sel,3e6+j.LA.values,np.where(fill>0,1e6+fill,j.L.values))),4)
    # (b) per-hand blends
    for w in (0.2,0.35,0.5,0.65,0.8):
        out[f'rankblend_w{w}']=round(ap((1-w)*rk(j.L.values)+w*rk(Q)),4)
        out[f'rankblendA_w{w}']=round(ap((1-w)*rk(j.LA.values)+w*rk(Q)),4)
    for b in (0.25,0.5,1.0,2.0):
        out[f'logitmix_b{b}']=round(ap(lg(j.L.values)+b*lg(np.clip(Q,1e-6,1-1e-6))),4)
    # (c) blend inside the A block only
    for tau in (0.3,0.4):
        for w in (0.3,0.5):
            sel=j.LA.values>=tau; blended=(1-w)*rk(j.LA.values)+w*rk(Q)
            fill=(Q>np.quantile(Q,0.8))*(2.0-trel)
            out[f'hybridblend_tau{tau}_w{w}']=round(ap(np.where(sel,3e6+blended,np.where(fill>0,1e6+fill,j.L.values))),4)
    res[key]=out; print(key,json.dumps({k:v for k,v in out.items() if not k.startswith('q_')}),flush=True); print('   q columns alone:',{k[2:]:v for k,v in out.items() if k.startswith('q_')},flush=True)
avg={k:round(float(np.mean([res[a][k] for a in res])),4) for k in res['di<-so'] if isinstance(res['di<-so'][k],float) and all(k in res[a] for a in res)}
print('AVERAGE',json.dumps(dict(sorted(avg.items(),key=lambda kv:-kv[1])[:14])))
json.dump({**res,'AVERAGE':avg},open(f'{O}/r5/m1_qblend.json','w'),indent=1)
