"""R5-E22: the decoder's last unchecked assumption - CONDITIONAL INDEPENDENCE.
It computes P(#A before h <= 4) as a Poisson-binomial over the per-hand p_A. The
generator, though, plants a SET of hands per pair, so the true count may be far less
dispersed than a Poisson-binomial with the same mean (sd ~ sqrt(sum p(1-p)) ~ 1.8 for DT).
Replace the cutoff by a one-parameter family with the same mean but tunable dispersion:
    L_A(h) = p_A(h) * sigma((4.5 - m_A(h)) / tau),  m_A(h) = sum_{j<h} p_A(j)
    L_B(h) = p_B(h) * sigma((4.5 - (M_A_total - p_A(h)) - m_B(h)) / tau)
tau -> 0 is a deterministic count; tau ~ 1.8 reproduces the Poisson-binomial for DT.
First: measure the OBSERVED dispersion of |A| on dev against the predicted one."""
import numpy as np, pandas as pd, sys, os, glob
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18'); sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
import ci_censored_event as C
from g4_decode import decode
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; R5=f'{O}/r5'
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9))); sg=lambda x:1/(1+np.exp(-x))
DEP=['lgb_base','lgb_sym','lgb_deep','lgb_shal','lgb_goss','lgb_dart','lgb_extra','lgb_col3','lgb_kern','lgb_gplay','lgb_role','lgb_l2','lgb_bag','cat']
def soft_decode(s,pa,pb,tau,K=5):
    LA=np.zeros(len(s)); LB=np.zeros(len(s))
    sl=s.slot.values
    for _,g in s.groupby('slot',sort=False):
        ix=g.index.values; a=pa[ix]; b=pb[ix]
        cum_a=np.concatenate([[0.0],np.cumsum(a)[:-1]]); cum_b=np.concatenate([[0.0],np.cumsum(b)[:-1]])
        tot_a=a.sum()
        LA[ix]=a*sg(((K-0.5)-cum_a)/tau)
        LB[ix]=b*sg(((K-0.5)-(tot_a-a)-cum_b)/tau)
    return LA+LB,LA,LB
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
    trueA=s.groupby('slot').isA.sum()
    pa0=sg(np.mean([lg(Z[n][0][0]) for n in nms],axis=0))
    sumA=pd.Series(pa0,index=s.slot.values).groupby(level=0).sum()
    varA=pd.Series(pa0*(1-pa0),index=s.slot.values).groupby(level=0).sum()
    print(f'=== {FAM}: observed |A| mean {trueA.mean():.2f} sd {trueA.std():.2f}  |  Poisson-binomial predicted mean {sumA.mean():.2f} sd {np.sqrt(varA).mean():.2f}')
    ref=[C.pair_ap(c,sc(to_c(decode(s,np.clip(sg(np.mean([lg(Z[n][si][0]) for n in nms],axis=0)),1e-9,1-1e-6),np.clip(sg(np.mean([lg(Z[n][si][1]) for n in nms],axis=0))*GB,1e-9,1-1e-6))[0])),counts).reindex(idx) for si in range(NS)]
    print(f'   exact Poisson-binomial decoder (deployed): {np.mean([a.mean() for a in ref]):.4f}')
    store={}
    for tau in (0.3,0.6,1.0,1.5,2.0,3.0,5.0):
        aps=[]
        for si in range(NS):
            pa=sg(np.mean([lg(Z[n][si][0]) for n in nms],axis=0)); pb=sg(np.mean([lg(Z[n][si][1]) for n in nms],axis=0))
            L,_,_=soft_decode(s,np.clip(pa,1e-9,1-1e-6),np.clip(pb*GB,1e-9,1-1e-6),tau)
            aps.append(C.pair_ap(c,sc(to_c(L)),counts).reindex(idx))
        store[tau]=aps
        d=np.mean([aps[i].values-ref[i].values for i in range(NS)],axis=0)
        pm=pd.Series(d,index=pool).groupby(level=0).mean(); bs=np.array([pm.values[rng.integers(0,len(pm),len(pm))].mean() for _ in range(2000)])
        print(f'   soft cutoff tau={tau:4.1f}: {np.mean([a.mean() for a in aps]):.4f}  d={d.mean():+.4f} poolP={float((bs>0).mean()):.3f}  seeds ' + ' '.join(f'{a.mean():.4f}' for a in aps),flush=True)
