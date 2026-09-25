"""R5-E16: the p_B scale is fitted at DEV exposure (~121 co-seated hands per pair) but
deployed at EVAL exposure (~86). Both the model's sum(p_B) and the true event count
scale with n, so the optimal FACTOR should be exposure-invariant - but that is an
assumption. Check it directly: optimal gb within each exposure tercile."""
import numpy as np, pandas as pd, sys, os, glob
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18'); sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
import ci_censored_event as C
from g4_decode import decode
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; R5=f'{O}/r5'
FAM='directed_transfer'; ab='di'
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9))); sg=lambda x:1/(1+np.exp(-x))
s=pd.read_parquet(f'{R5}/L/rows_{ab}.parquet').reset_index(drop=True); counts=s.groupby('slot').ev.sum()
cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
hmap=pd.read_parquet(f'{O}/np/hand_index.parquet').set_index('hand_id').hi
t45=pd.read_parquet(f'{O}/r3/t45_known_e_rerank.parquet'); t45['h']=t45.hand_id.map(hmap)
c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev']],on=['slot','h'],validate='one_to_one').merge(t45[['slot','h','tab']],on=['slot','h'],how='left').reset_index(drop=True)
mi=pd.MultiIndex.from_arrays([c.slot,c.h]); fmi=pd.MultiIndex.from_arrays([s.slot,s.h]); to_c=lambda v: pd.Series(v,index=fmi).reindex(mi).values
files=sorted(glob.glob(f'{R5}/L/{ab}__*.npy')); nms=[os.path.basename(f).split('__')[1][:-4] for f in files]
OLD=[n for n in nms if not n.startswith('x_')]; Z={n:np.load(f) for n,f in zip(nms,files)}; NS=Z[nms[0]].shape[0]
TAB=lg(c.tab.values); idx=np.array(counts.index)
nh=s.groupby('slot').size().reindex(idx); q1,q2=np.percentile(nh,[33,67])
strata={'low n (<=%.0f, closest to eval)'%q1:nh[nh<=q1].index,'mid':nh[(nh>q1)&(nh<=q2)].index,'high n (>%.0f)'%q2:nh[nh>q2].index,'ALL':idx}
print(f'dev n per pair: mean {nh.mean():.0f} median {nh.median():.0f}; eval frame is ~86')
print('gb    ' + '  '.join(f'{k[:22]:>22s}' for k in strata))
res={}
for gb in (1.0,1.2,1.5,1.8,2.2,3.0):
    aps=[]
    for si in range(NS):
        pa=sg(np.mean([lg(Z[n][si][0]) for n in OLD],axis=0)); pb=sg(np.mean([lg(Z[n][si][1]) for n in OLD],axis=0))
        L,_,_=decode(s,np.clip(pa,1e-9,1-1e-6),np.clip(pb*gb,1e-9,1-1e-6))
        aps.append(C.pair_ap(c,TAB+1.0*lg(to_c(L)),counts).reindex(idx))
    row=[float(np.mean([a.reindex(v).mean() for a in aps])) for v in strata.values()]
    res[gb]=row; print(f'{gb:4.1f}  ' + '  '.join(f'{v:22.4f}' for v in row),flush=True)
print('argmax gb per stratum:', {k:max(res,key=lambda g:res[g][i]) for i,k in enumerate(strata)})
