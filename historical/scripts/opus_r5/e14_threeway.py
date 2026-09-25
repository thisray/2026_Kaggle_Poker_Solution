"""R5-E14: the deployed forms only ever combine TWO of the three candidate-level signals.
DT uses TabICL + decoder L (no R15); SP uses R15 + L (no TabICL); CI uses TabICL + L.
Sweep a coarse simplex over the within-pair rank percentiles of all three."""
import numpy as np, pandas as pd, sys, os, json, glob
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18'); sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
import ci_censored_event as C
from g4_decode import decode
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; R5=f'{O}/r5'
FAM=os.environ['FAM']; ab=FAM[:2]; GB=float(os.environ.get('GB','1.0'))
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9))); sg=lambda x:1/(1+np.exp(-x))
s=pd.read_parquet(f'{R5}/L/rows_{ab}.parquet').reset_index(drop=True); counts=s.groupby('slot').ev.sum()
cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
hmap=pd.read_parquet(f'{O}/np/hand_index.parquet').set_index('hand_id').hi
t45=pd.read_parquet(f'{O}/r3/t45_known_e_rerank.parquet'); t45['h']=t45.hand_id.map(hmap)
c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev']],on=['slot','h'],validate='one_to_one').merge(t45[['slot','h','tab']],on=['slot','h'],how='left').reset_index(drop=True)
mi=pd.MultiIndex.from_arrays([c.slot,c.h]); fmi=pd.MultiIndex.from_arrays([s.slot,s.h]); to_c=lambda v: pd.Series(v,index=fmi).reindex(mi).values
files=sorted(glob.glob(f'{R5}/L/{ab}__*.npy')); nms=[os.path.basename(f).split('__')[1][:-4] for f in files]
OLD=[n for n in nms if not n.startswith('x_')]; Z={n:np.load(f) for n,f in zip(nms,files)}; NS=Z[nms[0]].shape[0]
rk=lambda v: pd.Series(v).groupby(c.slot.values).rank(pct=True).values
TABr=rk(c.tab.values); R15r=rk(-c.r.values.astype(float)); idx=np.array(counts.index); pool=idx//900
Ls=[]
for si in range(NS):
    pa=sg(np.mean([lg(Z[n][si][0]) for n in OLD],axis=0)); pb=sg(np.mean([lg(Z[n][si][1]) for n in OLD],axis=0))
    Ls.append(to_c(decode(s,np.clip(pa,1e-9,1-1e-6),np.clip(pb*GB,1e-9,1-1e-6))[0]))
Lr=[rk(x) for x in Ls]
DEPfn={'directed_transfer':lambda si: lg(c.tab.values)+1.0*lg(Ls[si]),
       'soft_play':lambda si: 0.65*R15r+0.35*Lr[si],
       'coordinated_isolation':lambda si: lg(c.tab.values)+3.0*lg(Ls[si])}[FAM]
dep=[C.pair_ap(c,DEPfn(si),counts).reindex(idx) for si in range(NS)]
dm=float(np.mean([a.mean() for a in dep])); rng=np.random.default_rng(7)
print(f'=== {FAM} (GB={GB}): current best form = {dm:.4f}')
rows=[]
for wl in np.arange(0.2,0.85,0.1):
    for wt in np.arange(0.0,1.0-wl+1e-9,0.1):
        wr=1-wl-wt
        if wr<-1e-9: continue
        aps=[C.pair_ap(c,wl*Lr[si]+wt*TABr+wr*R15r,counts).reindex(idx) for si in range(NS)]
        m=float(np.mean([a.mean() for a in aps]))
        d=np.mean([aps[i].values-dep[i].values for i in range(NS)],axis=0)
        pm=pd.Series(d,index=pool).groupby(level=0).mean(); bs=np.array([pm.values[rng.integers(0,len(pm),len(pm))].mean() for _ in range(2000)])
        rows.append((round(wl,2),round(wt,2),round(wr,2),m,d.mean(),float((bs>0).mean()),[a.mean() for a in aps]))
rows.sort(key=lambda r:-r[3])
for wl,wt,wr,m,d,p,seeds in rows[:10]:
    print(f'  wL={wl:.1f} wTAB={wt:.1f} wR15={wr:.1f}  {m:.4f}  d={d:+.4f} poolP={p:.3f}  seeds ' + ' '.join(f'{x:.4f}' for x in seeds))
