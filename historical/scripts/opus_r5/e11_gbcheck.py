"""R5-E11: stress-test the p_B rescaling gain on DT. The Poisson-binomial decoder
consumes ABSOLUTE probabilities; the B model is trained with right-censoring so it
estimates the B-EVENT rate, and the decoder needs P(#B before h <= remaining slots).
Check: (a) is the response in the scale smooth and peaked (a calibration) or monotone
(a degenerate 'earliest B wins' rule)? (b) is the optimum the same in all 3 seeds and
in both halves of the pairs? (c) how does it compare with R4's deployed g4sym number?"""
import numpy as np, pandas as pd, sys, os, json, glob
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18'); sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
import ci_censored_event as C
from g4_decode import decode
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; R5=f'{O}/r5'
FAM=os.environ.get('FAM','directed_transfer'); ab=FAM[:2]
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9))); sg=lambda x:1/(1+np.exp(-x))
s=pd.read_parquet(f'{R5}/L/rows_{ab}.parquet').reset_index(drop=True); counts=s.groupby('slot').ev.sum()
cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
hmap=pd.read_parquet(f'{O}/np/hand_index.parquet').set_index('hand_id').hi
t45=pd.read_parquet(f'{O}/r3/t45_known_e_rerank.parquet'); t45['h']=t45.hand_id.map(hmap)
c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev']],on=['slot','h'],validate='one_to_one').merge(t45[['slot','h','tab']],on=['slot','h'],how='left').reset_index(drop=True)
mi=pd.MultiIndex.from_arrays([c.slot,c.h]); fmi=pd.MultiIndex.from_arrays([s.slot,s.h]); to_c=lambda v: pd.Series(v,index=fmi).reindex(mi).values
files=sorted(glob.glob(f'{R5}/L/{ab}__*.npy')); nms=[os.path.basename(f).split('__')[1][:-4] for f in files]
OLD=[n for n in nms if not n.startswith('x_')]
Z={n:np.load(f) for n,f in zip(nms,files)}; NS=Z[nms[0]].shape[0]
TAB=lg(c.tab.values); R15=-c.r.values.astype(float); rk=lambda v: pd.Series(v).groupby(c.slot.values).rank(pct=True).values
FORM={'directed_transfer':('stack',1.0),'coordinated_isolation':('stack',3.0),'soft_play':('rank',0.35)}[FAM]
def score(v):
    return TAB+FORM[1]*lg(v) if FORM[0]=='stack' else (1-FORM[1])*rk(R15)+FORM[1]*rk(v)
idx=np.array(counts.index); pool=idx//900
print(f'=== {FAM}: {len(OLD)} old configs, form {FORM}')
print(f'{"gb":>5s} ' + ' '.join(f'seed{i}' for i in range(NS)) + '   mean    poolP(vs gb=1)   halfA   halfB')
base=None; rows={}
for gb in (1.0,0.7,1.2,1.5,1.8,2.2,3.0,5.0,10.0):
    aps=[]
    for si in range(NS):
        pa=sg(np.mean([lg(Z[n][si][0]) for n in OLD],axis=0)); pb=sg(np.mean([lg(Z[n][si][1]) for n in OLD],axis=0))
        L,_,_=decode(s,np.clip(pa,1e-9,1-1e-6),np.clip(pb*gb,1e-9,1-1e-6))
        aps.append(C.pair_ap(c,score(to_c(L)),counts).reindex(idx))
    rows[gb]=aps; m=float(np.mean([a.mean() for a in aps]))
    if gb==1.0: base=aps
    d=np.mean([aps[i].values-base[i].values for i in range(NS)],axis=0)
    pm=pd.Series(d,index=pool).groupby(level=0).mean(); rng=np.random.default_rng(7)
    bs=np.array([pm.values[rng.integers(0,len(pm),len(pm))].mean() for _ in range(4000)])
    hA=idx[::2]; hB=idx[1::2]
    ma=float(np.mean([a.reindex(hA).mean() for a in aps])); mb=float(np.mean([a.reindex(hB).mean() for a in aps]))
    print(f'{gb:5.1f} ' + ' '.join(f'{a.mean():.4f}' for a in aps) + f'   {m:.4f}   {float((bs>0).mean()):.3f}          {ma:.4f} {mb:.4f}')
# reference points
sym=[Z['lgb_sym'][si][2] for si in range(NS)]
print(f'reference lgb_sym|{FORM[0]}{FORM[1]}: {np.mean([C.pair_ap(c,score(to_c(x)),counts).mean() for x in sym]):.4f}')
print(f'reference R15: {C.pair_ap(c,R15,counts).mean():.4f}')
