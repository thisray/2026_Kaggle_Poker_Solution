"""R5-E21: the two-type decoder scores a hand as L = L_A + L_B, an implicit 1:1 weight
between the two event types. That weight has never been tuned. It is NOT the same as
rescaling p_B (which changes the decoder's internal counts); this changes only the final
mixture. Sweep lambda in L = L_A + lambda*L_B, at the deployed p_B scale and at 1.0."""
import numpy as np, pandas as pd, sys, os, json
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18'); sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
import ci_censored_event as C
from g4_decode import decode
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; R5=f'{O}/r5'
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9))); sg=lambda x:1/(1+np.exp(-x))
DEP=['lgb_base','lgb_sym','lgb_deep','lgb_shal','lgb_goss','lgb_dart','lgb_extra','lgb_col3','lgb_kern','lgb_gplay','lgb_role','lgb_l2','lgb_bag','cat']
for FAM,ab,GB,form in (('directed_transfer','di',1.5,'stack1'),('soft_play','so',1.0,'rank035')):
    s=pd.read_parquet(f'{R5}/L/rows_{ab}.parquet').reset_index(drop=True); counts=s.groupby('slot').ev.sum()
    cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
    hmap=pd.read_parquet(f'{O}/np/hand_index.parquet').set_index('hand_id').hi
    t45=pd.read_parquet(f'{O}/r3/t45_known_e_rerank.parquet'); t45['h']=t45.hand_id.map(hmap)
    c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev']],on=['slot','h'],validate='one_to_one').merge(t45[['slot','h','tab']],on=['slot','h'],how='left').reset_index(drop=True)
    mi=pd.MultiIndex.from_arrays([c.slot,c.h]); fmi=pd.MultiIndex.from_arrays([s.slot,s.h]); to_c=lambda v: pd.Series(v,index=fmi).reindex(mi).values
    Z={n:np.load(f'{R5}/L/{ab}__{n}.npy') for n in DEP if os.path.exists(f'{R5}/L/{ab}__{n}.npy')}
    nms=list(Z); NS=Z[nms[0]].shape[0]
    rk=lambda v: pd.Series(v).groupby(c.slot.values).rank(pct=True).values
    TAB=lg(c.tab.values); R15=-c.r.values.astype(float); idx=np.array(counts.index); pool=idx//900; rng=np.random.default_rng(7)
    sc=(lambda L: TAB+1.0*lg(np.clip(L,1e-9,None))) if form=='stack1' else (lambda L: 0.65*rk(R15)+0.35*rk(L))
    store={}
    for gb in (1.0,GB):
        for lam in (0.0,0.3,0.5,0.7,1.0,1.4,2.0,3.0):
            aps=[]
            for si in range(NS):
                pa=sg(np.mean([lg(Z[n][si][0]) for n in nms],axis=0)); pb=sg(np.mean([lg(Z[n][si][1]) for n in nms],axis=0))
                _,LA,LBx=decode(s,np.clip(pa,1e-9,1-1e-6),np.clip(pb*gb,1e-9,1-1e-6))
                aps.append(C.pair_ap(c,sc(to_c(LA+lam*LBx)),counts).reindex(idx))
            store[(gb,lam)]=aps
    ref=store[(GB,1.0)]
    print(f'=== {FAM} (form {form}); reference = deployed gb={GB}, lambda=1: {np.mean([a.mean() for a in ref]):.4f}')
    print(f'{"gb":>4s} ' + '  '.join(f'lam={l}' for l in (0.0,0.3,0.5,0.7,1.0,1.4,2.0,3.0)))
    for gb in (1.0,GB):
        print(f'{gb:4.1f} ' + '  '.join(f'{np.mean([a.mean() for a in store[(gb,l)]]):.4f}' for l in (0.0,0.3,0.5,0.7,1.0,1.4,2.0,3.0)),flush=True)
    bk=max(store,key=lambda k: np.mean([a.mean() for a in store[k]]))
    d=np.mean([store[bk][i].values-ref[i].values for i in range(NS)],axis=0)
    pm=pd.Series(d,index=pool).groupby(level=0).mean(); bs=np.array([pm.values[rng.integers(0,len(pm),len(pm))].mean() for _ in range(4000)])
    print(f'  best {bk}: {np.mean([a.mean() for a in store[bk]]):.4f}  d={d.mean():+.4f} poolP={float((bs>0).mean()):.3f}  seeds ' + ' '.join(f'{a.mean():.4f}' for a in store[bk]))
