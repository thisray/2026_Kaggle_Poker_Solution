"""R5-E18: measure EXACTLY the config set that g6z_deploy_zoo.py deploys (the 14
original normal-view configs), at gb=1.5, in the deployed form. Earlier sweeps used a
name filter that accidentally also admitted two flipped-view models; the fusion is
equal-weight so the numbers barely move, but the deployed number must be the one that
was measured."""
import numpy as np, pandas as pd, sys, os, glob
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18'); sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
import ci_censored_event as C
from g4_decode import decode
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; R5=f'{O}/r5'
DEPLOYED=['lgb_base','lgb_sym','lgb_deep','lgb_shal','lgb_goss','lgb_dart','lgb_extra','lgb_col3','lgb_kern','lgb_gplay','lgb_role','lgb_l2','lgb_bag','cat']
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9))); sg=lambda x:1/(1+np.exp(-x))
for FAM,ab,GB,form in (('directed_transfer','di',1.5,'stack1'),('soft_play','so',1.0,'rank035')):
    s=pd.read_parquet(f'{R5}/L/rows_{ab}.parquet').reset_index(drop=True); counts=s.groupby('slot').ev.sum()
    cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
    hmap=pd.read_parquet(f'{O}/np/hand_index.parquet').set_index('hand_id').hi
    t45=pd.read_parquet(f'{O}/r3/t45_known_e_rerank.parquet'); t45['h']=t45.hand_id.map(hmap)
    c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev']],on=['slot','h'],validate='one_to_one').merge(t45[['slot','h','tab']],on=['slot','h'],how='left').reset_index(drop=True)
    mi=pd.MultiIndex.from_arrays([c.slot,c.h]); fmi=pd.MultiIndex.from_arrays([s.slot,s.h]); to_c=lambda v: pd.Series(v,index=fmi).reindex(mi).values
    Z={n:np.load(f'{R5}/L/{ab}__{n}.npy') for n in DEPLOYED if os.path.exists(f'{R5}/L/{ab}__{n}.npy')}
    NS=Z[DEPLOYED[0]].shape[0]; nms=list(Z)
    rk=lambda v: pd.Series(v).groupby(c.slot.values).rank(pct=True).values
    TAB=lg(c.tab.values); R15=-c.r.values.astype(float); idx=np.array(counts.index); pool=idx//900
    sc=(lambda L: TAB+1.0*lg(L)) if form=='stack1' else (lambda L: 0.65*rk(R15)+0.35*rk(L))
    def go(gb):
        return [C.pair_ap(c,sc(to_c(decode(s,np.clip(sg(np.mean([lg(Z[n][si][0]) for n in nms],axis=0)),1e-9,1-1e-6),np.clip(sg(np.mean([lg(Z[n][si][1]) for n in nms],axis=0))*gb,1e-9,1-1e-6))[0])),counts).reindex(idx) for si in range(NS)]
    a1=go(1.0); ag=go(GB)
    sym=[C.pair_ap(c,sc(to_c(Z['lgb_sym'][si][2])),counts).reindex(idx) for si in range(NS)]
    r15=C.pair_ap(c,R15,counts).reindex(idx)
    rng=np.random.default_rng(7)
    def cmp(x,y):
        d=np.mean([x[i].values-y[i].values for i in range(NS)],axis=0); pm=pd.Series(d,index=pool).groupby(level=0).mean()
        bs=np.array([pm.values[rng.integers(0,len(pm),len(pm))].mean() for _ in range(4000)]); return d.mean(),float((bs>0).mean())
    d1,p1=cmp(ag,a1); d2,p2=cmp(ag,sym); d3,p3=cmp(ag,[r15]*NS)
    print(f'=== {FAM}  exactly {len(nms)} deployed configs, form {form}, gb={GB}')
    print(f'   R15                    {r15.mean():.4f}')
    print(f'   single lgb_sym         {np.mean([a.mean() for a in sym]):.4f}')
    print(f'   {len(nms)}-config fusion gb=1.0 {np.mean([a.mean() for a in a1]):.4f}')
    print(f'   {len(nms)}-config fusion gb={GB} {np.mean([a.mean() for a in ag]):.4f}   seeds ' + ' '.join(f'{a.mean():.4f}' for a in ag))
    print(f'      vs gb=1.0  d={d1:+.4f} poolP={p1:.3f} | vs lgb_sym d={d2:+.4f} poolP={p2:.3f} | vs R15 d={d3:+.4f} poolP={p3:.3f}',flush=True)
