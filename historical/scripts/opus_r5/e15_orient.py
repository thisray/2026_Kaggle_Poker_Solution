"""R5-E15: soft_play's A-type event is direction-SYMMETRIC ('either member folds the
better hand to the other'), but the in-family event models score one orientation only -
the role features are assigned by a net-chip-flow heuristic that is meaningless for a
symmetric family. The transfer experiments always score BOTH orientations and OR them,
p = 1-(1-p1)(1-p2); the in-family models never have. Test the OR-combination of the
normal-view and flipped-view zoo fusions against the normal-view fusion."""
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
OLD=[n for n in nms if not n.startswith('x_')]; FLIP=[n for n in nms if n.endswith('_flip')]
Z={n:np.load(f) for n,f in zip(nms,files)}; NS=Z[nms[0]].shape[0]
rk=lambda v: pd.Series(v).groupby(c.slot.values).rank(pct=True).values
TAB=lg(c.tab.values); R15r=rk(-c.r.values.astype(float)); idx=np.array(counts.index); pool=idx//900
form={'directed_transfer':lambda L: TAB+1.0*lg(L),'coordinated_isolation':lambda L: TAB+3.0*lg(L),'soft_play':lambda L: 0.65*R15r+0.35*rk(L)}[FAM]
def fuse(sub,si,k): return sg(np.mean([lg(Z[n][si][k]) for n in sub],axis=0))
def ev(name,mk):
    aps=[]
    for si in range(NS):
        pa,pb=mk(si); L,_,_=decode(s,np.clip(pa,1e-9,1-1e-6),np.clip(pb*GB,1e-9,1-1e-6)); aps.append(C.pair_ap(c,form(to_c(L)),counts).reindex(idx))
    return name,float(np.mean([a.mean() for a in aps])),aps
print(f'=== {FAM} (GB={GB}): {len(OLD)} normal-view configs, {len(FLIP)} flipped-view configs')
R=[ev('normal fusion',lambda si:(fuse(OLD,si,0),fuse(OLD,si,1)))]
if FLIP:
    R.append(ev('flip fusion',lambda si:(fuse(FLIP,si,0),fuse(FLIP,si,1))))
    R.append(ev('OR(normal,flip)',lambda si:(1-(1-fuse(OLD,si,0))*(1-fuse(FLIP,si,0)),1-(1-fuse(OLD,si,1))*(1-fuse(FLIP,si,1)))))
    R.append(ev('MAX(normal,flip)',lambda si:(np.maximum(fuse(OLD,si,0),fuse(FLIP,si,0)),np.maximum(fuse(OLD,si,1),fuse(FLIP,si,1)))))
    R.append(ev('MEAN(normal,flip)',lambda si:(0.5*(fuse(OLD,si,0)+fuse(FLIP,si,0)),0.5*(fuse(OLD,si,1)+fuse(FLIP,si,1)))))
    R.append(ev('logit-avg(normal,flip)',lambda si:(fuse(OLD+FLIP,si,0),fuse(OLD+FLIP,si,1))))
base=R[0][2]; rng=np.random.default_rng(7)
for nm,m,aps in R:
    d=np.mean([aps[i].values-base[i].values for i in range(NS)],axis=0)
    pm=pd.Series(d,index=pool).groupby(level=0).mean(); bs=np.array([pm.values[rng.integers(0,len(pm),len(pm))].mean() for _ in range(4000)])
    print(f'  {nm:24s} {m:.4f}  d={d.mean():+.4f} poolP={float((bs>0).mean()):.3f}  seeds ' + ' '.join(f'{a.mean():.4f}' for a in aps))
