"""R5-S1: end-to-end LOCAL score with the OFFICIAL metric on dev, for whole candidate
CONFIGURATIONS rather than one component at a time. Every round so far has extrapolated
additively from component deltas; R3 skipped this because a dev family-classification OOF
did not exist - R5-d6 built one (99.45% accurate).
Builds a dev-side submission (risk = pair-model OOF, predicted_behavior = family OOF
argmax, evidence = each family's deployed picker) over the exposure-matched devsub11
population, and scores it with src/pokerlab/metrics.score_submission.
NOTE: dev has no other_coordination, so this measures the known-family part only."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, os, json, glob
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker/src')
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917')
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18')
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
from pokerlab.metrics import stable_ap, evidence_ap5
import ci_censored_event as C
from g4_decode import decode
from pairfeat import build
from pairfeat2 import build2
from sklearn.model_selection import GroupKFold
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; R5=f'{O}/r5'
FAMS=['directed_transfer','soft_play','coordinated_isolation']
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9))); sg=lambda x:1/(1+np.exp(-x))
# ---------- risk: two rankings ----------
T=pd.read_parquet(f'{O}/m5_both_train_oof.parquet')
SUB='devsub11'; d=T[T.src==SUB].copy()
hid=set(T[(T.src==SUB)&(T.label==-1)&(T.oof>0.3)].key); d=d[~d.key.isin(hid)].sort_values('key',kind='mergesort').reset_index(drop=True)
RISK={'m5_both(single, stands in for the deployed 3-model average)':d.oof.values}
for f in ('m15_o_pos_a_train_oof.parquet','m15_o_goss_b_train_oof.parquet'):
    if os.path.exists(f'{O}/{f}'):
        t2=pd.read_parquet(f'{O}/{f}'); t2=t2[t2.src==SUB].set_index('key').oof
        RISK.setdefault('_fuse',[]).append(lg(t2.reindex(d.key).values))
if '_fuse' in RISK:
    fu=RISK.pop('_fuse'); RISK['equal-weight fusion of 3 pair models']=sg(np.mean([lg(d.oof.values)]+fu,axis=0))
t=pd.read_parquet(f'{O}/ptab_{SUB}.parquet'); t['key']=t.p_lo*12000+t.p_hi; t=t.set_index('key').loc[d.key].reset_index()
X=pd.concat([build(t),build2(t)],axis=1); pos=d.y.values==1; fam=d.fam.values
yf=pd.Series(fam).map({f:i for i,f in enumerate(FAMS)}).values; prob=np.zeros((len(d),3))
for f in range(5):
    tr=pos&(d.fold.values!=f); va=d.fold.values==f
    m=lgb.train(dict(objective='multiclass',num_class=3,learning_rate=.05,num_leaves=15,min_data_in_leaf=10,feature_fraction=.6,verbose=-1,num_threads=6),lgb.Dataset(X[tr],yf[tr].astype(int)),300)
    prob[va]=m.predict(X[va])
beh=np.array(FAMS)[prob.argmax(1)]
# ---------- evidence: two configurations ----------
cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
hmap=pd.read_parquet(f'{O}/np/hand_index.parquet').set_index('hand_id').hi; inv={v:k for k,v in hmap.items()}
t45=pd.read_parquet(f'{O}/r3/t45_known_e_rerank.parquet'); t45['h']=t45.hand_id.map(hmap)
EV={}
for FAM in FAMS:
    ab=FAM[:2]; s=pd.read_parquet(f'{R5}/L/rows_{ab}.parquet').reset_index(drop=True)
    c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev']],on=['slot','h'],validate='one_to_one').merge(t45[['slot','h','tab']],on=['slot','h'],how='left').reset_index(drop=True)
    fmi=pd.MultiIndex.from_arrays([s.slot,s.h]); mi=pd.MultiIndex.from_arrays([c.slot,c.h]); to_c=lambda v: pd.Series(v,index=fmi).reindex(mi).values
    files=sorted(glob.glob(f'{R5}/L/{ab}__*.npy')); nms=[os.path.basename(f).split('__')[1][:-4] for f in files]
    OLD=[n for n in nms if not n.startswith('x_')]; Z={n:np.load(f) for n,f in zip(nms,files)}; NS=Z[nms[0]].shape[0]
    rkf=lambda v: pd.Series(v).groupby(c.slot.values).rank(pct=True).values
    TAB=lg(c.tab.values); R15=-c.r.values.astype(float)
    GB={'directed_transfer':1.5}.get(FAM,1.0)
    Ls=[to_c(decode(s,np.clip(sg(np.mean([lg(Z[n][si][0]) for n in OLD],axis=0)),1e-9,1-1e-6),np.clip(sg(np.mean([lg(Z[n][si][1]) for n in OLD],axis=0))*GB,1e-9,1-1e-6))[0]) for si in range(NS)]
    new={'directed_transfer':lambda si: TAB+1.0*lg(Ls[si]),'soft_play':lambda si: 0.65*rkf(R15)+0.35*rkf(Ls[si]),'coordinated_isolation':lambda si: TAB+3.0*lg(Ls[si])}[FAM]
    for tag,sc in (('r10_ci_config',lambda si: R15),('r32_config',new)):
        for si in range(NS):
            z=c[['slot','h','ts']].copy(); z['s']=sc(si)
            z=z.sort_values(['slot','s','ts','h'],ascending=[True,False,True,True],kind='stable')
            z['rk']=z.groupby('slot').cumcount()+1; z=z[z.rk<=5]
            EV.setdefault((tag,si),{}).update({int(sl):[inv[int(x)] for x in g.h] for sl,g in z.groupby('slot')})
# ---------- assemble and score with the official metric ----------
pl=pd.read_parquet(f'{O}/player_local_v1.parquet').set_index('player_gi')
lo=(d.key.values//12000); hi=(d.key.values%12000)
slot=(pl.pool.reindex(lo).values*900+pl.local.reindex(lo).values*30+pl.local.reindex(hi).values).astype(int)
truth_ev={}
tv=pd.read_parquet(f'{O}/t5_dev_seq.parquet').rename(columns={'sl':'slot'})
tv=tv[tv.ev.astype(bool)]
for sl,g in tv.groupby('slot'): truth_ev[int(sl)]=[inv[int(x)] for x in g.h]
print(f'population {len(d)}  positives {int(pos.sum())}  slots with truth evidence {len(truth_ev)}',flush=True)
out={}
for rname,risk in RISK.items():
    P=stable_ap(pos.astype(int),risk)
    B=float(np.mean([stable_ap((fam==f).astype(int),risk*(beh==f)) for f in FAMS]))
    for tag in ('r10_ci_config','r32_config'):
        Es=[]
        for si in range(3):
            vals=[evidence_ap5(EV[(tag,si)].get(int(sl),[]),truth_ev.get(int(sl),[])) for sl in slot[pos]]
            Es.append(float(np.mean(vals)))
        E=float(np.mean(Es))
        out[f'{rname} | {tag}']=dict(P=round(P,5),E=round(E,5),B=round(B,5),S=round(0.7*P+0.2*E+0.1*B,5))
        print(f'{rname[:46]:46s} {tag:14s} P={P:.5f} E={E:.5f} B={B:.5f}  S={0.7*P+0.2*E+0.1*B:.5f}',flush=True)
json.dump(out,open(f'{R5}/s1_end2end.json','w'),indent=1)
