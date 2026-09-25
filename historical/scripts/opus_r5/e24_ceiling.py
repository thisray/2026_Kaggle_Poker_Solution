"""R5-E24: is the evidence score at ITS OWN Bayes limit? If the decoder's listing
probabilities L were the true posteriors, the EXPECTED AP@5 of ranking by them is
computable by Monte-Carlo (draw a truth set from Bernoulli(L) per hand, score our fixed
top-5 against it). Compare with the ACHIEVED AP@5 against the real truth.
  achieved ~ expected -> the model is calibrated and we are at the limit of the
      information it has; only NEW information can help (confirms R3's Round-20 claim
      quantitatively).
  achieved < expected -> the probabilities are over-confident; there is noise the model
      does not know about, and a better-calibrated or stronger model could still gain."""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18'); sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
import ci_censored_event as C
from g4_decode import decode
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; R5=f'{O}/r5'
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9))); sg=lambda x:1/(1+np.exp(-x))
DEP=['lgb_base','lgb_sym','lgb_deep','lgb_shal','lgb_goss','lgb_dart','lgb_extra','lgb_col3','lgb_kern','lgb_gplay','lgb_role','lgb_l2','lgb_bag','cat']
rng=np.random.default_rng(11)
for FAM,ab,GB in (('directed_transfer','di',1.5),('soft_play','so',1.0),('coordinated_isolation','co',1.0)):
    s=pd.read_parquet(f'{R5}/L/rows_{ab}.parquet').reset_index(drop=True); counts=s.groupby('slot').ev.sum()
    cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
    hmap=pd.read_parquet(f'{O}/np/hand_index.parquet').set_index('hand_id').hi
    t45=pd.read_parquet(f'{O}/r3/t45_known_e_rerank.parquet'); t45['h']=t45.hand_id.map(hmap)
    c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev']],on=['slot','h'],validate='one_to_one').merge(t45[['slot','h','tab']],on=['slot','h'],how='left').reset_index(drop=True)
    mi=pd.MultiIndex.from_arrays([c.slot,c.h]); fmi=pd.MultiIndex.from_arrays([s.slot,s.h]); to_c=lambda v: pd.Series(v,index=fmi).reindex(mi).values
    Z={n:np.load(f'{R5}/L/{ab}__{n}.npy') for n in DEP if os.path.exists(f'{R5}/L/{ab}__{n}.npy')}
    nms=list(Z); NS=Z[nms[0]].shape[0]
    rk=lambda v: pd.Series(v).groupby(c.slot.values).rank(pct=True).values
    TAB=lg(c.tab.values); R15=-c.r.values.astype(float)
    sc=(lambda L: 0.65*rk(R15)+0.35*rk(L)) if FAM=='soft_play' else (lambda L: TAB+(3.0 if FAM=='coordinated_isolation' else 1.0)*lg(np.clip(L,1e-9,None)))
    ach=[]; exp=[]; calib=[]
    for si in range(NS):
        pa=sg(np.mean([lg(Z[n][si][0]) for n in nms],axis=0)); pb=sg(np.mean([lg(Z[n][si][1]) for n in nms],axis=0))
        L,_,_=decode(s,np.clip(pa,1e-9,1-1e-6),np.clip(pb*GB,1e-9,1-1e-6))
        Lc=to_c(L); score=sc(Lc)
        ach.append(C.pair_ap(c,score,counts).mean())
        # calibration: predicted number of listed hands per pair (over the FULL frame) vs the true count
        pred_n=pd.Series(L,index=s.slot.values).groupby(level=0).sum(); calib.append((pred_n.mean(),counts.mean()))
        # our fixed top-5 per pair under this score
        z=c[['slot','h','ts']].copy(); z['s']=score
        z=z.sort_values(['slot','s','ts','h'],ascending=[True,False,True,True],kind='stable'); z['rk']=z.groupby('slot').cumcount()+1
        top=z[z.rk<=5]; picks={sl:list(g.h) for sl,g in top.groupby('slot')}
        Lfull=pd.Series(np.clip(L,0,1),index=pd.MultiIndex.from_arrays([s.slot,s.h]))
        vals=[]
        for sl,g in Lfull.groupby(level=0):
            p=g.values; hs=g.index.get_level_values(1).values; pk=picks.get(sl,[])
            pos={h_:i for i,h_ in enumerate(hs)}
            pi=np.array([p[pos[h_]] if h_ in pos else 0.0 for h_ in pk])
            for _ in range(60):
                truth=rng.random(len(p))<p
                k=truth.sum()
                if k==0: vals.append(0.0); continue
                hit=rng.random(len(pk))<pi   # whether each pick is in the sampled truth (consistent marginal)
                hits=0; tot=0.0
                for i,hh in enumerate(hit):
                    if hh: hits+=1; tot+=hits/(i+1)
                vals.append(tot/min(5,k))
        exp.append(float(np.mean(vals)))
    print(f'=== {FAM}: achieved AP@5 {np.mean(ach):.4f}   expected-under-own-probabilities {np.mean(exp):.4f}   ratio {np.mean(ach)/np.mean(exp):.3f}')
    print(f'    calibration of the listing count: predicted {calib[0][0]:.2f} listed hands/pair vs true {calib[0][1]:.2f}',flush=True)
