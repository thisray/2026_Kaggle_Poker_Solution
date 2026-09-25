"""R5-E10: final sweep over the full event zoo (original 14 + 17 new: flipped
orientation view, cross-family typing, XGBoost, CatBoost-deep, more hyper-parameters).
Three axes, all evaluated with the R4-G4 protocol (pool GroupKFold x 3 seeds, official
AP@5 inside the frozen R15 top-20, full-truth denominators):
  1. equal-weight probability-level fusion over config subsets (no fitted weights)
  2. a p_A / p_B rescaling before the decoder - the Poisson-binomial consumes ABSOLUTE
     probabilities, and R5-d3 showed rescaling helps the decoder alone; it was never
     combined with the TabICL stack, which is what is deployed
  3. deployment form: alone / logit stack with TabICL / R15 rank blend
Reported against the DEPLOYED reference for each family, with a pool bootstrap."""
import numpy as np, pandas as pd, sys, os, json, glob
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18'); sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
import ci_censored_event as C
from g4_decode import decode
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; R5=f'{O}/r5'
FAM=os.environ['FAM']; ab=FAM[:2]
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9))); sg=lambda x:1/(1+np.exp(-x))
s=pd.read_parquet(f'{R5}/L/rows_{ab}.parquet').reset_index(drop=True); counts=s.groupby('slot').ev.sum()
cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
hmap=pd.read_parquet(f'{O}/np/hand_index.parquet').set_index('hand_id').hi
t45=pd.read_parquet(f'{O}/r3/t45_known_e_rerank.parquet'); t45['h']=t45.hand_id.map(hmap)
c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev']],on=['slot','h'],validate='one_to_one').merge(t45[['slot','h','tab']],on=['slot','h'],how='left').reset_index(drop=True)
mi=pd.MultiIndex.from_arrays([c.slot,c.h]); fmi=pd.MultiIndex.from_arrays([s.slot,s.h]); to_c=lambda v: pd.Series(v,index=fmi).reindex(mi).values
files=sorted(glob.glob(f'{R5}/L/{ab}__*.npy')); nms=[os.path.basename(f).split('__')[1][:-4] for f in files]
Z={n:np.load(f) for n,f in zip(nms,files)}; NS=Z[nms[0]].shape[0]
TAB=lg(c.tab.values); R15=-c.r.values.astype(float); rk=lambda v: pd.Series(v).groupby(c.slot.values).rank(pct=True).values
OLD=[n for n in nms if not n.startswith('x_')]; NEW=[n for n in nms if n.startswith('x_')]
FLIP=[n for n in nms if n.endswith('_flip')]; XF=[n for n in nms if n.endswith('_xf')]
GROUPS={'old14':OLD,'new':NEW,'all':nms,'old+flip':OLD+FLIP,'old+xf':OLD+XF,'noview':[n for n in nms if n not in ('lgb_kern','lgb_gplay','lgb_role')]}
print(f'=== {FAM}: {len(nms)} configs ({len(OLD)} old, {len(NEW)} new: {len(FLIP)} flip, {len(XF)} xfam), {len(counts)} pairs',flush=True)
per={}; res={'R15':float(C.pair_ap(c,R15,counts).mean())}
def add(nm,Ls):
    forms={'alone':[to_c(x) for x in Ls]}
    for b in (0.5,1.0,2.0,3.0): forms[f'stack{b}']=[TAB+b*lg(to_c(x)) for x in Ls]
    for w in (0.35,0.5): forms[f'rank{w}']=[(1-w)*rk(R15)+w*rk(to_c(x)) for x in Ls]
    for fn,vs in forms.items():
        aps=[C.pair_ap(c,v,counts) for v in vs]; res[f'{nm}|{fn}']=float(np.mean([a.mean() for a in aps])); per[f'{nm}|{fn}']=aps
# deployed reference
Lsym=[Z['lgb_sym'][si][2] for si in range(NS)]; add('lgb_sym',Lsym)
pa0=[sg(np.mean([lg(Z[n][si][0]) for n in nms],axis=0)) for si in range(NS)]
pb0=[sg(np.mean([lg(Z[n][si][1]) for n in nms],axis=0)) for si in range(NS)]
for gname,sub in GROUPS.items():
    if len(sub)<2: continue
    pa=[sg(np.mean([lg(Z[n][si][0]) for n in sub],axis=0)) for si in range(NS)]
    pb=[sg(np.mean([lg(Z[n][si][1]) for n in sub],axis=0)) for si in range(NS)]
    for ga,gb in ((1.0,1.0),(1.0,1.5),(1.0,2.0),(1.2,1.0),(0.8,1.0),(1.2,2.0)):
        Ls=[decode(s,np.clip(pa[si]*ga,1e-9,1-1e-6),np.clip(pb[si]*gb,1e-9,1-1e-6))[0] for si in range(NS)]
        add(f'PF_{gname}_g{ga}_{gb}',Ls)
DEP={'directed_transfer':'lgb_sym|stack1.0','soft_play':'R15','coordinated_isolation':'lgb_sym|stack3.0'}[FAM]
if DEP=='R15': per['R15']=[C.pair_ap(c,R15,counts)]*NS
top=sorted(res.items(),key=lambda kv:-kv[1])
print(json.dumps({k:round(v,4) for k,v in top[:14]},indent=1))
print(f'DEPLOYED reference {DEP} = {res[DEP]:.4f}')
pool=np.array(counts.index)//900; rng=np.random.default_rng(7)
best=[]
for k,v in top[:40]:
    if k==DEP or k not in per: continue
    d=np.mean([per[k][si].values-per[DEP][si].values for si in range(NS)],axis=0)
    pm=pd.Series(d,index=pool).groupby(level=0).mean(); bs=np.array([pm.values[rng.integers(0,len(pm),len(pm))].mean() for _ in range(4000)])
    best.append((k,res[k],d.mean(),float((bs>0).mean())))
for k,v,d,p in best[:14]: print(f'   {k:34s} {v:.4f}  d={d:+.4f}  poolP={p:.3f}')
json.dump({k:round(v,5) for k,v in res.items()},open(f'{R5}/e10_final_{ab}.json','w'),indent=1)
