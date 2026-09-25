"""R5-K1: where exactly do the 1.2 missed picks per pair sit? Per-slot hit rate and
per-TYPE recall for the deployed configuration. If the loss is concentrated in the
B-type slots it is a timing problem (B events are plentiful, only the earliest are
listed); if in A it is a detection problem."""
import numpy as np, pandas as pd, sys, os, glob
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18'); sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
import ci_censored_event as C
from g4_decode import decode
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; R5=f'{O}/r5'
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9))); sg=lambda x:1/(1+np.exp(-x))
cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
hmap=pd.read_parquet(f'{O}/np/hand_index.parquet').set_index('hand_id').hi
t45=pd.read_parquet(f'{O}/r3/t45_known_e_rerank.parquet'); t45['h']=t45.hand_id.map(hmap)
for fam,ab in (('directed_transfer','di'),('soft_play','so'),('coordinated_isolation','co')):
    s=pd.read_parquet(f'{R5}/L/rows_{ab}.parquet').reset_index(drop=True); counts=s.groupby('slot').ev.sum()
    c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev','isA','isB']],on=['slot','h'],validate='one_to_one').merge(t45[['slot','h','tab']],on=['slot','h'],how='left').reset_index(drop=True)
    fmi=pd.MultiIndex.from_arrays([s.slot,s.h]); mi=pd.MultiIndex.from_arrays([c.slot,c.h]); to_c=lambda v: pd.Series(v,index=fmi).reindex(mi).values
    files=sorted(glob.glob(f'{R5}/L/{ab}__*.npy')); nms=[os.path.basename(f).split('__')[1][:-4] for f in files]
    Z={n:np.load(f) for n,f in zip(nms,files)}; NS=Z[nms[0]].shape[0]
    Lsym=to_c(np.mean([Z['lgb_sym'][si][2] for si in range(NS)],axis=0))
    pa=sg(np.mean([lg(Z[n][si][0]) for n in nms for si in range(NS)],axis=0)); pb=sg(np.mean([lg(Z[n][si][1]) for n in nms for si in range(NS)],axis=0))
    Lp=to_c(decode(s,np.clip(pa,1e-9,1-1e-6),np.clip(pb,1e-9,1-1e-6))[0])
    DEP={'directed_transfer':lg(c.tab.values)+1.0*lg(Lsym),'soft_play':-c.r.values.astype(float),'coordinated_isolation':lg(c.tab.values)+3.0*lg(Lp)}[fam]
    z=c[['slot','h','ts','ev','isA','isB']].copy(); z['s']=DEP
    z=z.sort_values(['slot','s','ts','h'],ascending=[True,False,True,True],kind='stable'); z['rk']=z.groupby('slot').cumcount()+1
    top=z[z.rk<=5]
    print(f'== {fam}: pairs {len(counts)}  true A {int(s.isA.sum())} B {int(s.isB.sum())}')
    print('   slot hit rate  :', {int(k):round(float(v),3) for k,v in top.groupby('rk').ev.mean().items()})
    print('   picks by type  :', {'A':int(top.isA.sum()),'B':int(top.isB.sum()),'wrong':int((~top.ev.astype(bool)).sum())})
    ra=s.isA.sum(); rb=s.isB.sum()
    print(f'   RECALL         : A {int(top.isA.sum())}/{int(ra)} = {top.isA.sum()/max(ra,1):.3f}   B {int(top.isB.sum())}/{int(rb)} = {top.isB.sum()/max(rb,1):.3f}')
    # what do the wrong picks look like relative to the missed truths?
    miss=z[(z.rk>5)&z.ev.astype(bool)]
    print(f'   missed truths  : A {int(miss.isA.sum())} B {int(miss.isB.sum())}  (their median deployed rank {miss.rk.median():.0f})')
