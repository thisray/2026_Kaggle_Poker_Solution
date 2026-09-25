"""R5-E12: the p_B rescaling win says the decoder is sensitive to the ABSOLUTE scale of
its inputs. Push that: (a) a wider p_A grid at the winning gb; (b) a PER-PAIR
normalisation (set each pair's sum(p) to a target, which is a proper calibration rather
than one global constant); (c) a power transform p^gamma (changes the shape, not just
the scale). Same protocol; report per-seed and pool bootstrap against gb=1.5 alone."""
import numpy as np, pandas as pd, sys, os, json, glob
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18'); sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
import ci_censored_event as C
from g4_decode import decode
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; R5=f'{O}/r5'
FAM=os.environ.get('FAM','directed_transfer'); ab=FAM[:2]; GB0=float(os.environ.get('GB0','1.5'))
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9))); sg=lambda x:1/(1+np.exp(-x))
s=pd.read_parquet(f'{R5}/L/rows_{ab}.parquet').reset_index(drop=True); counts=s.groupby('slot').ev.sum()
cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
hmap=pd.read_parquet(f'{O}/np/hand_index.parquet').set_index('hand_id').hi
t45=pd.read_parquet(f'{O}/r3/t45_known_e_rerank.parquet'); t45['h']=t45.hand_id.map(hmap)
c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev']],on=['slot','h'],validate='one_to_one').merge(t45[['slot','h','tab']],on=['slot','h'],how='left').reset_index(drop=True)
mi=pd.MultiIndex.from_arrays([c.slot,c.h]); fmi=pd.MultiIndex.from_arrays([s.slot,s.h]); to_c=lambda v: pd.Series(v,index=fmi).reindex(mi).values
files=sorted(glob.glob(f'{R5}/L/{ab}__*.npy')); nms=[os.path.basename(f).split('__')[1][:-4] for f in files]
OLD=[n for n in nms if not n.startswith('x_')]; Z={n:np.load(f) for n,f in zip(nms,files)}; NS=Z[nms[0]].shape[0]
TAB=lg(c.tab.values); R15=-c.r.values.astype(float); rk=lambda v: pd.Series(v).groupby(c.slot.values).rank(pct=True).values
FORM={'directed_transfer':('stack',1.0),'coordinated_isolation':('stack',3.0),'soft_play':('rank',0.35)}[FAM]
score=lambda v: TAB+FORM[1]*lg(v) if FORM[0]=='stack' else (1-FORM[1])*rk(R15)+FORM[1]*rk(v)
idx=np.array(counts.index); pool=idx//900; sl=s.slot.values
PA=[sg(np.mean([lg(Z[n][si][0]) for n in OLD],axis=0)) for si in range(NS)]
PB=[sg(np.mean([lg(Z[n][si][1]) for n in OLD],axis=0)) for si in range(NS)]
def run(pa_list,pb_list):
    return [C.pair_ap(c,score(to_c(decode(s,np.clip(pa_list[si],1e-9,1-1e-6),np.clip(pb_list[si],1e-9,1-1e-6))[0])),counts).reindex(idx) for si in range(NS)]
def norm_to(p,T):
    ssum=pd.Series(p).groupby(sl).transform('sum').values
    return p*T/np.maximum(ssum,1e-9)
base=run(PA,[p*GB0 for p in PB]); bm=float(np.mean([a.mean() for a in base]))
print(f'=== {FAM}  reference = global gb={GB0}: {bm:.4f}  (per-seed ' + ' '.join(f'{a.mean():.4f}' for a in base) + ')')
rng=np.random.default_rng(7)
def rep(nm,aps):
    m=float(np.mean([a.mean() for a in aps])); d=np.mean([aps[i].values-base[i].values for i in range(NS)],axis=0)
    pm=pd.Series(d,index=pool).groupby(level=0).mean(); bs=np.array([pm.values[rng.integers(0,len(pm),len(pm))].mean() for _ in range(4000)])
    print(f'  {nm:34s} {m:.4f}  d={d.mean():+.4f} poolP={float((bs>0).mean()):.3f}  seeds ' + ' '.join(f'{a.mean():.4f}' for a in aps),flush=True)
    return m
for ga in (0.6,0.8,1.2,1.5,2.0):
    rep(f'ga={ga} (gb={GB0})',run([p*ga for p in PA],[p*GB0 for p in PB]))
for T in (3.0,4.0,5.0,6.0):
    rep(f'perpair sumA={T} (gb={GB0})',run([norm_to(p,T) for p in PA],[p*GB0 for p in PB]))
for T in (6.0,9.0,12.0,16.0,24.0):
    rep(f'perpair sumB={T}',run(PA,[norm_to(p,T) for p in PB]))
for g in (0.6,0.8,1.25,1.6):
    rep(f'pB^{g} (renorm to gb={GB0} mean)',run(PA,[np.clip(p,1e-9,1)**g*GB0 for p in PB]))
