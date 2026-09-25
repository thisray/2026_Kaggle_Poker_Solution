"""R5 d3: is the typed decoder hurt by miscalibrated p_A / p_B?
The Poisson-binomial decoder consumes ABSOLUTE probabilities (P(#A before h <= 4)),
so a systematic scale error in p_A shifts every listing probability.
Diagnostic: per-pair sum(p) vs the true per-pair count. Then re-decode after a
monotone recalibration and re-score with the official AP@5 protocol."""
import numpy as np, pandas as pd, sys, os, json
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18')
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
import ci_censored_event as C
from g4_decode import decode
from sklearn.isotonic import IsotonicRegression
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'
TAG=os.environ.get('TAG','g4sym')
FAMS={'di':'directed_transfer','so':'soft_play','co':'coordinated_isolation'}
cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
hmap=pd.read_parquet(f'{O}/np/hand_index.parquet').set_index('hand_id').hi
t45=pd.read_parquet(f'{O}/r3/t45_known_e_rerank.parquet'); t45['h']=t45.hand_id.map(hmap)
lg=lambda p: np.log(np.clip(p,1e-5,1-1e-5)/(1-np.clip(p,1e-5,1-1e-5)))
res={}
for ab,fam in FAMS.items():
    f=f'{O}/r4/{TAG}_oof_{ab}.npy'
    if not os.path.exists(f): f=f'{O}/r4/g4_oof_{ab}.npy'; rowf=f'{O}/r4/g4_rows_{ab}.parquet'
    else: rowf=f'{O}/r4/{TAG}_rows_{ab}.parquet'
    A=np.load(f); s=pd.read_parquet(rowf).reset_index(drop=True)
    counts=s.groupby('slot').ev.sum()
    c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev']],on=['slot','h'],validate='one_to_one').merge(t45[['slot','h','tab']],on=['slot','h'],how='left').reset_index(drop=True)
    mi=pd.MultiIndex.from_arrays([c.slot,c.h])
    trueA=s.groupby('slot').isA.sum(); trueB=s.groupby('slot').isB.sum()
    out={}
    print(f'=== {fam}  rows {len(s)} pairs {len(counts)}  true |A| mean {trueA.mean():.3f} |B| mean {trueB.mean():.3f}',flush=True)
    for si in range(A.shape[0]):
        pA,pB,L=A[si]
        sumA=pd.Series(pA,index=s.slot.values).groupby(level=0).sum(); sumB=pd.Series(pB,index=s.slot.values).groupby(level=0).sum()
        if si==0:
            print(f'  seed0  sum(pA) mean {sumA.mean():.3f} (true {trueA.mean():.3f}); sum(pB) mean {sumB.mean():.3f} (true {trueB.mean():.3f})')
            print(f'         corr(sumA,trueA) {np.corrcoef(sumA.reindex(trueA.index),trueA)[0,1]:.3f}  corr(sumB,trueB) {np.corrcoef(sumB.reindex(trueB.index),trueB)[0,1]:.3f}')
        def score(nm,Lx):
            v=pd.Series(Lx,index=pd.MultiIndex.from_arrays([s.slot,s.h])).reindex(mi).values
            out.setdefault(nm,[]).append(float(C.pair_ap(c,v,counts).mean()))
            out.setdefault(nm+'|stack1',[]).append(float(C.pair_ap(c,lg(c.tab.values)+1.0*lg(np.clip(v,1e-9,1)),counts).mean()))
        score('base',L)
        # global multiplicative recalibration of pA and pB (grid)
        best=None
        for ga in (0.5,0.7,0.85,1.0,1.2,1.5,2.0,3.0):
            for gb in (0.5,0.7,0.85,1.0,1.2,1.5,2.0,3.0):
                a=np.clip(pA*ga,0,1-1e-6); b=np.clip(pB*gb,0,1-1e-6)
                Lx,_,_=decode(s,a,b)
                v=pd.Series(Lx,index=pd.MultiIndex.from_arrays([s.slot,s.h])).reindex(mi).values
                ap=float(C.pair_ap(c,v,counts).mean())
                out.setdefault(f'g{ga}_{gb}',[]).append(ap)
                if best is None or ap>best[0]: best=(ap,ga,gb)
        if si==0: print(f'  seed0 best global scale {best}',flush=True)
    res[fam]={k:round(float(np.mean(v)),4) for k,v in out.items()}
    top=sorted(res[fam].items(),key=lambda kv:-kv[1])[:8]
    print(f'  {fam} base={res[fam]["base"]:.4f} base|stack1={res[fam]["base|stack1"]:.4f}  top: {top}',flush=True)
json.dump(res,open(f'{O}/r5/d3_calib_{TAG}.json','w'),indent=1)
