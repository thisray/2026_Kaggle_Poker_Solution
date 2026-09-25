"""R5-F6: the only lane with real headroom left is the fourth family, and its
bottleneck is transfer efficiency (0.71). The 14-config zoo raised soft_play IN-family
by +0.011; does it raise TRANSFER? Leave-one-family-out, exact hybrid slate
([transferred A with L_A>=tau by L_A] + [target's own B-rule by time] + [rest by L]),
single config vs probability-level zoo fusion of the source family."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, os, json
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18'); sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
import ci_censored_event as C
from g4_decode import decode, runs
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; NJ=int(os.environ.get('NJ',6))
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9))); sg=lambda x:1/(1+np.exp(-x))
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
CFG=[('base',dict(C.PARAMS)),('deep',{**C.PARAMS,'num_leaves':63,'min_child_samples':20,'n_estimators':500,'learning_rate':.03}),
 ('shal',{**C.PARAMS,'num_leaves':7,'n_estimators':900,'learning_rate':.02,'min_child_samples':60}),
 ('goss',{**C.PARAMS,'boosting_type':'goss','num_leaves':31,'n_estimators':500}),
 ('extra',{**C.PARAMS,'extra_trees':True,'num_leaves':31,'n_estimators':600}),
 ('col3',{**C.PARAMS,'colsample_bytree':.3,'num_leaves':31,'n_estimators':600}),
 ('l2',{**C.PARAMS,'reg_lambda':200,'num_leaves':31,'n_estimators':600}),
 ('bag',{**C.PARAMS,'subsample':.6,'subsample_freq':1,'num_leaves':31,'n_estimators':600}),
 ('dart',{**C.PARAMS,'boosting_type':'dart','n_estimators':400,'learning_rate':.06,'drop_rate':.1})]
def brule(d,fam):
    if fam=='directed_transfer': return (((d.k_ps_eq_last<=0.3)&(d.x_conS>=5)&(d.k_pr_won>0))|((d.k_pr_eq_last<=0.3)&(d.x_conR>=5)&(d.k_ps_won>0))).values
    return ((d.sd_any.astype(bool))&(d.both_flop.astype(bool))&(d.x_s_aggr_post==0)&(d.x_r_aggr_post==0)).values
res={}
for tgt,src in (('directed_transfer',['soft_play']),('soft_play',['directed_transfer'])):
    key=f'{tgt[:2]}<-{src[0][:2]}'; m=(sA.fam==tgt).values; tA=sA[m].reset_index(drop=True); tF=sF[m].reset_index(drop=True)
    ss,iA,iB=types(sA[sA.fam.isin(src)].reset_index(drop=True))
    BR=brule(tA,tgt).astype(float); trel=tA.groupby('slot').ts.rank(pct=True).values; fill=BR*(2.0-trel)
    out={}
    def slate(LA,L,tau): 
        sel=LA>=tau; return np.where(sel,3e6+LA,np.where(BR>0,1e6+fill,L))
    lA=[];lB=[]
    for nm,par in CFG:
        for sd in (27,7,99):
            a=both(lgb.LGBMClassifier(**{**par,'n_jobs':NJ,'random_state':sd}).fit(ss.loc[iA,FS].astype(float),ss.loc[iA,'isA']),tA,tF)
            b=both(lgb.LGBMClassifier(**{**par,'n_jobs':NJ,'random_state':sd}).fit(ss.loc[iB,FS].astype(float),ss.loc[iB,'isB']),tA,tF)
            lA.append(lg(a)); lB.append(lg(b))
            if nm=='base' and sd==27:
                L0,LA0,_=decode(tA,np.clip(a,1e-6,1-1e-6),np.clip(b,1e-6,1-1e-6))
                for tau in (0.3,0.4,0.5): out[f'single_tau{tau}']=round(ap(tA,slate(LA0,L0,tau)),4)
        print(f'  {key} {nm} done',flush=True)
    pa=np.clip(sg(np.mean(lA,axis=0)),1e-6,1-1e-6); pb=np.clip(sg(np.mean(lB,axis=0)),1e-6,1-1e-6)
    Lz,LAz,_=decode(tA,pa,pb)
    for tau in (0.3,0.4,0.5,0.6): out[f'zoo_tau{tau}']=round(ap(tA,slate(LAz,Lz,tau)),4)
    out['zoo_plainL']=round(ap(tA,Lz),4); out['zoo_nA_tau0.3']=round(float(pd.Series((LAz>=0.3).astype(int)).groupby(tA.slot.values).sum().mean()),2)
    out['zoo_nA_tau0.4']=round(float(pd.Series((LAz>=0.4).astype(int)).groupby(tA.slot.values).sum().mean()),2)
    res[key]=out; print(key,json.dumps(out),flush=True)
avg={k:round(float(np.mean([res[a][k] for a in res])),4) for k in res['di<-so'] if all(k in res[a] for a in res)}
print('AVERAGE',json.dumps(avg)); json.dump({**res,'AVERAGE':avg},open(f'{O}/r5/f6_zoo_transfer.json','w'),indent=1)
