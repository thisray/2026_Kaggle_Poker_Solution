"""R5-E3: fuse the event zoo at the PROBABILITY level (average logit p_A and
logit p_B over configs, then run the two-type decoder once), not at the L level.
L is a Poisson-binomial functional of the whole pair's probability vector, so
averaging L across models breaks the decoder's internal consistency; averaging
the inputs does not. Equal weights only - no weight fitting (the P lane showed
fitted weights lose). Deployment forms: alone, logit stack with TabICL, and the
R15 rank blend. Paired pool-level bootstrap against the deployed reference."""
import numpy as np, pandas as pd, sys, os, json, glob
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18'); sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
import ci_censored_event as C
from g4_decode import decode
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; R5=f'{O}/r5'
FAM=os.environ['FAM']; ab=FAM[:2]
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9))); sg=lambda x: 1/(1+np.exp(-x))
s=pd.read_parquet(f'{R5}/L/rows_{ab}.parquet').reset_index(drop=True); counts=s.groupby('slot').ev.sum()
cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
hmap=pd.read_parquet(f'{O}/np/hand_index.parquet').set_index('hand_id').hi
t45=pd.read_parquet(f'{O}/r3/t45_known_e_rerank.parquet'); t45['h']=t45.hand_id.map(hmap)
c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev']],on=['slot','h'],validate='one_to_one').merge(t45[['slot','h','tab']],on=['slot','h'],how='left').reset_index(drop=True)
mi=pd.MultiIndex.from_arrays([c.slot,c.h]); fmi=pd.MultiIndex.from_arrays([s.slot,s.h])
to_c=lambda v: pd.Series(v,index=fmi).reindex(mi).values
files=sorted(glob.glob(f'{R5}/L/{ab}__*.npy')); names=[os.path.basename(f).split('__')[1][:-4] for f in files]
Z={n:np.load(f) for n,f in zip(names,files)}; NS=next(iter(Z.values())).shape[0]
TAB=lg(c.tab.values); R15=-c.r.values
GROUPS={'ALL':names,'noview':[n for n in names if n not in ('lgb_kern','lgb_gplay','lgb_role')],
        'core':[n for n in names if n in ('lgb_sym','lgb_shal','lgb_goss','lgb_extra','lgb_col3','lgb_deep','lgb_bag','lgb_l2','cat')]}
per={}; res={'R15':float(C.pair_ap(c,R15,counts).mean())}
def add(nm,Lseeds):
    forms={'alone':[to_c(x) for x in Lseeds]}
    for b in (0.5,1.0,2.0,3.0,4.0): forms[f'stack{b}']=[TAB+b*lg(to_c(x)) for x in Lseeds]
    for w in (0.35,0.5,0.65,0.8):
        forms[f'rank{w}']=[ (1-w)*pd.Series(R15).groupby(c.slot.values).rank(pct=True).values + w*pd.Series(to_c(x)).groupby(c.slot.values).rank(pct=True).values for x in Lseeds]
    for fn,vs in forms.items():
        aps=[C.pair_ap(c,v,counts) for v in vs]; res[f'{nm}|{fn}']=float(np.mean([a.mean() for a in aps])); per[f'{nm}|{fn}']=aps
for n in names: add(n,[Z[n][si][2] for si in range(NS)])
for gname,sub in GROUPS.items():
    if len(sub)<2: continue
    Ls=[]
    for si in range(NS):
        pa=sg(np.mean([lg(Z[n][si][0]) for n in sub],axis=0)); pb=sg(np.mean([lg(Z[n][si][1]) for n in sub],axis=0))
        Ls.append(decode(s,np.clip(pa,1e-9,1-1e-6),np.clip(pb,1e-9,1-1e-6))[0])
    add(f'PFUSE_{gname}',Ls)
    add(f'LFUSE_{gname}',[sg(np.mean([lg(Z[n][si][2]) for n in sub],axis=0)) for si in range(NS)])
top=sorted(res.items(),key=lambda kv:-kv[1])
print(f'=== {FAM}: {len(names)} configs, {len(counts)} pairs'); print(json.dumps({k:round(v,4) for k,v in top[:20]},indent=1))
json.dump({k:round(v,5) for k,v in res.items()},open(f'{R5}/e3_fuse_{ab}.json','w'),indent=1)
REF=os.environ.get('REF','')
pool=(np.array(counts.index)//900)
for candname in [k for k in res if k.startswith('PFUSE_') or k.startswith('LFUSE_')]+[top[0][0]]:
    if REF not in per or candname==REF: continue
    d=np.mean([per[candname][si].values-per[REF][si].values for si in range(NS)],axis=0)
    dfp=pd.Series(d,index=pool); pm=dfp.groupby(level=0).mean(); rng=np.random.default_rng(7)
    bs=np.array([pm.values[rng.integers(0,len(pm),len(pm))].mean() for _ in range(4000)])
    print(f'{candname:34s} {res[candname]:.4f}  vs {REF} {res[REF]:.4f}  d={d.mean():+.4f} poolP(>0)={float((bs>0).mean()):.3f}')
