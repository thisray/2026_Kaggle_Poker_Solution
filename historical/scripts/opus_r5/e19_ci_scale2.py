"""R5-E19: the DT win came from rescaling a right-censored probability before a
Poisson-binomial decoder. CI's deployed patch uses the SAME kind of decoder
(first_k_marginal over an untyped, right-censored event model with the rich feature
set) - R5-e13 only swept the scale on the 29-feature base model. Sweep it on the full
zoo fusion (which for CI is effectively the rich untyped model, since CI has almost no
type-B events so L = L_A = first_k_marginal(p_A))."""
import numpy as np, pandas as pd, sys, os, glob
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18'); sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
import ci_censored_event as C
from g4_decode import decode
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; R5=f'{O}/r5'
FAM='coordinated_isolation'; ab='co'
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9))); sg=lambda x:1/(1+np.exp(-x))
s=pd.read_parquet(f'{R5}/L/rows_{ab}.parquet').reset_index(drop=True); counts=s.groupby('slot').ev.sum()
cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
hmap=pd.read_parquet(f'{O}/np/hand_index.parquet').set_index('hand_id').hi
t45=pd.read_parquet(f'{O}/r3/t45_known_e_rerank.parquet'); t45['h']=t45.hand_id.map(hmap)
c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev']],on=['slot','h'],validate='one_to_one').merge(t45[['slot','h','tab']],on=['slot','h'],how='left').reset_index(drop=True)
mi=pd.MultiIndex.from_arrays([c.slot,c.h]); fmi=pd.MultiIndex.from_arrays([s.slot,s.h]); to_c=lambda v: pd.Series(v,index=fmi).reindex(mi).values
DEP=['lgb_base','lgb_sym','lgb_deep','lgb_shal','lgb_goss','lgb_dart','lgb_extra','lgb_col3','lgb_kern','lgb_gplay','lgb_role','lgb_l2','lgb_bag','cat']
Z={n:np.load(f'{R5}/L/{ab}__{n}.npy') for n in DEP if os.path.exists(f'{R5}/L/{ab}__{n}.npy')}
nms=list(Z); NS=Z[nms[0]].shape[0]
TAB=lg(c.tab.values); idx=np.array(counts.index); pool=idx//900; rng=np.random.default_rng(7)
print(f'=== {FAM}: {len(nms)} configs; sum(p_A) per pair median {pd.Series(Z[nms[0]][0][0]).groupby(s.slot.values).sum().median():.2f}, true evidence/pair {counts.mean():.2f}')
print(f'{"ga":>5s} ' + '  '.join(f'b={b}' for b in (1.0,2.0,3.0,4.0,5.0)))
store={}
for ga in (0.5,0.7,0.85,1.0,1.2,1.5,2.0,3.0):
    row=[]
    for b in (1.0,2.0,3.0,4.0,5.0):
        aps=[]
        for si in range(NS):
            pa=sg(np.mean([lg(Z[n][si][0]) for n in nms],axis=0)); pb=sg(np.mean([lg(Z[n][si][1]) for n in nms],axis=0))
            L,_,_=decode(s,np.clip(pa*ga,1e-9,1-1e-6),np.clip(pb,1e-9,1-1e-6))
            aps.append(C.pair_ap(c,TAB+b*lg(to_c(L)),counts).reindex(idx))
        m=float(np.mean([a.mean() for a in aps])); row.append(m); store[(ga,b)]=aps
    print(f'{ga:5.2f} ' + '  '.join(f'{v:.4f}' for v in row),flush=True)
ref=store[(1.0,3.0)]
best=max(store,key=lambda k: np.mean([a.mean() for a in store[k]]))
d=np.mean([store[best][i].values-ref[i].values for i in range(NS)],axis=0)
pm=pd.Series(d,index=pool).groupby(level=0).mean(); bs=np.array([pm.values[rng.integers(0,len(pm),len(pm))].mean() for _ in range(4000)])
print(f'best (ga,beta)={best}: {np.mean([a.mean() for a in store[best]]):.4f}  vs (1.0,3.0) {np.mean([a.mean() for a in ref]):.4f}  d={d.mean():+.4f} poolP={float((bs>0).mean()):.3f}')
print(f'  per-seed at best: ' + ' '.join(f'{a.mean():.4f}' for a in store[best]) + '   at (1.0,3.0): ' + ' '.join(f'{a.mean():.4f}' for a in ref))
print(f'  NOTE R4 deployed CI patch measures 0.7503 on dev with its own richer pipeline; this sweep is internal-paired.')
