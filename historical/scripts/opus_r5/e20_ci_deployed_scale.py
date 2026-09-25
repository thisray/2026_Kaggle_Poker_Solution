"""R5-E20: sweep the input scale of the DEPLOYED CI pipeline (R4-x7): the saved rich
model OOF probabilities -> first_k_marginal -> logit(tab) + beta*logit(q) [+ hard filter].
Same mechanism that gave DT +0.0101. No retraining: x4_oofp_co.npy is saved."""
import numpy as np, pandas as pd, sys, json
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18')
import ci_censored_event as C
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'
lg=lambda p: np.log(np.clip(p,1e-5,1-1e-5)/(1-np.clip(p,1e-5,1-1e-5)))
hmap=pd.read_parquet(f'{O}/np/hand_index.parquet').set_index('hand_id').hi
t45=pd.read_parquet(f'{O}/r3/t45_known_e_rerank.parquet'); t45['h']=t45.hand_id.map(hmap)
cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
t5=pd.read_parquet(f'{O}/t5_dev_seq.parquet').rename(columns={'sl':'slot'})[['slot','h','y1','pa_at_trig']]
s=pd.read_parquet(f'{O}/r4/x4_rows_co.parquet').reset_index(drop=True); P=np.load(f'{O}/r4/x4_oofp_co.npy'); counts=s.groupby('slot').ev.sum()
c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts','y1','pa_at_trig'],errors='ignore').merge(s[['slot','h','ts','ev']],on=['slot','h'],validate='one_to_one')
c=c.merge(t45[['slot','h','tab','fold']],on=['slot','h'],how='left',validate='one_to_one').merge(t5,on=['slot','h'],how='left').reset_index(drop=True)
viol=(~((c.pa_at_trig==6)&c.y1.isin([2,3]))).values.astype(float)
mi=pd.MultiIndex.from_arrays([c.slot,c.h]); fmi=pd.MultiIndex.from_arrays([s.slot,s.h])
idx=counts.index; pl=pd.Series(np.asarray(idx)//900,index=idx); up=pl.unique(); rng=np.random.RandomState(4)
def pp(scs): return np.mean([C.pair_ap(c,sc,counts).values for sc in scs],axis=0)
def boot(d): return float(np.mean([np.mean(np.concatenate([pd.Series(d,index=idx)[pl==p_].values for p_ in rng.choice(up,len(up))]))>0 for _ in range(1000)]))
print(f'CI deployed pipeline: {P.shape[0]} seeds, sum(p) per pair median {pd.Series(P[0]).groupby(s.slot.values).sum().median():.2f}, true evidence/pair {counts.mean():.2f}')
base=None; res={}
print(f'{"ga":>5s} ' + '  '.join(f'b={b}{h}' for b in (2.5,3.0,4.0,6.0) for h in ('','H')))
for ga in (0.7,0.85,1.0,1.1,1.2,1.35,1.5,2.0):
    Q=[pd.Series(C.first_k_marginal(s,np.clip(p*ga,0,1)),index=fmi).reindex(mi).values for p in P]
    row=[]
    for b in (2.5,3.0,4.0,6.0):
        for hard in (0,1):
            v=pp([lg(c.tab.values)+b*lg(q)-1000*hard*viol for q in Q]); res[(ga,b,hard)]=v; row.append(v.mean())
    print(f'{ga:5.2f} ' + '  '.join(f'{x:.4f}' for x in row),flush=True)
base=res[(1.0,4.0,0)]   # the deployed configuration family (x7 nested picked beta=4, hard=0)
bk=max(res,key=lambda k: res[k].mean())
d=res[bk]-base
print(f'deployed-like (ga=1.0,beta=4,hard=0): {base.mean():.4f}')
print(f'best {bk}: {res[bk].mean():.4f}  d={d.mean():+.4f}  poolP={boot(d):.3f}')
for k in sorted(res,key=lambda k:-res[k].mean())[:6]:
    print(f'   {k}: {res[k].mean():.4f}  d={res[k].mean()-base.mean():+.4f} poolP={boot(res[k]-base):.3f}')
json.dump({str(k):float(v.mean()) for k,v in res.items()},open(f'{O}/r5/e20_ci_deployed_scale.json','w'),indent=1)
