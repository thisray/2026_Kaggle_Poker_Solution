"""R5-E25: R5-e24 shows the event probabilities are OVER-CONFIDENT (achieved AP@5 is
only 0.87-0.96 of what the model's own probabilities predict) while the per-pair listing
COUNT is well calibrated. That is a shape problem, not a mean problem - a logit
TEMPERATURE, not a scale. A monotone recalibration of p does not change p's ranking, but
it does change the decoder's output L (the Poisson-binomial is non-linear in p), so it
can change the final ranking. Never tested.
   p'_A = sigmoid(logit(p_A)/T_A),   p'_B = G_B * sigmoid(logit(p_B)/T_B)"""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18'); sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
import ci_censored_event as C
from g4_decode import decode
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; R5=f'{O}/r5'
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9))); sg=lambda x:1/(1+np.exp(-x))
DEP=['lgb_base','lgb_sym','lgb_deep','lgb_shal','lgb_goss','lgb_dart','lgb_extra','lgb_col3','lgb_kern','lgb_gplay','lgb_role','lgb_l2','lgb_bag','cat']
for FAM,ab,GB0 in (('directed_transfer','di',1.5),('soft_play','so',1.0),('coordinated_isolation','co',1.0)):
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
    sc=(lambda L: 0.65*rk(R15)+0.35*rk(L)) if FAM=='soft_play' else (lambda L: TAB+(3.0 if FAM=='coordinated_isolation' else 1.0)*lg(np.clip(L,1e-9,None)))
    PA=[sg(np.mean([lg(Z[n][si][0]) for n in nms],axis=0)) for si in range(NS)]
    PB=[sg(np.mean([lg(Z[n][si][1]) for n in nms],axis=0)) for si in range(NS)]
    def go(TA,TB,GB):
        aps=[]
        for si in range(NS):
            a=np.clip(sg(lg(PA[si])/TA),1e-9,1-1e-6); b=np.clip(GB*sg(lg(PB[si])/TB),1e-9,1-1e-6)
            L,_,_=decode(s,a,b); aps.append(C.pair_ap(c,sc(to_c(L)),counts).reindex(idx))
        return aps
    ref=go(1.0,1.0,GB0); rm=float(np.mean([a.mean() for a in ref]))
    print(f'=== {FAM}: deployed (T=1,T=1,G={GB0}) = {rm:.4f}')
    best=None; rows=[]
    for TA in (0.7,0.85,1.0,1.2,1.5):
        for TB in (0.7,0.85,1.0,1.2,1.5):
            for GB in ({1.0,GB0}):
                aps=go(TA,TB,GB); m=float(np.mean([a.mean() for a in aps]))
                rows.append((m,TA,TB,GB,aps))
                if best is None or m>best[0]: best=(m,TA,TB,GB,aps)
    rows.sort(key=lambda r:-r[0])
    for m,TA,TB,GB,aps in rows[:5]:
        d=np.mean([aps[i].values-ref[i].values for i in range(NS)],axis=0)
        pm=pd.Series(d,index=pool).groupby(level=0).mean(); bs=np.array([pm.values[rng.integers(0,len(pm),len(pm))].mean() for _ in range(4000)])
        print(f'   T_A={TA:4.2f} T_B={TB:4.2f} G_B={GB:3.1f}: {m:.4f}  d={d.mean():+.4f} poolP={float((bs>0).mean()):.3f}  seeds ' + ' '.join(f'{a.mean():.4f}' for a in aps),flush=True)
