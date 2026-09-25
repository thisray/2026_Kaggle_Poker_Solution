"""R5-E2: evaluate single configs and parameter-free fusions of the event-model zoo.
Protocol identical to R4-G4/G8: official AP@5 inside the frozen R15 top-20 with
full-truth denominators, averaged over the same 3 seeds. Pair bootstrap for the
fusion-vs-reference deltas."""
import numpy as np, pandas as pd, sys, os, json, glob, itertools
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18')
import ci_censored_event as C
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; R5=f'{O}/r5'
FAM=os.environ['FAM']; ab=FAM[:2]
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9)))
s=pd.read_parquet(f'{R5}/L/rows_{ab}.parquet').reset_index(drop=True)
counts=s.groupby('slot').ev.sum()
cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
hmap=pd.read_parquet(f'{O}/np/hand_index.parquet').set_index('hand_id').hi
t45=pd.read_parquet(f'{O}/r3/t45_known_e_rerank.parquet'); t45['h']=t45.hand_id.map(hmap)
c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev']],on=['slot','h'],validate='one_to_one').merge(t45[['slot','h','tab']],on=['slot','h'],how='left').reset_index(drop=True)
mi=pd.MultiIndex.from_arrays([c.slot,c.h]); full_mi=pd.MultiIndex.from_arrays([s.slot,s.h])
def to_c(v): return pd.Series(v,index=full_mi).reindex(mi).values
def ap(score): return C.pair_ap(c,score,counts)
files=sorted(glob.glob(f'{R5}/L/{ab}__*.npy')); names=[os.path.basename(f).split('__')[1][:-4] for f in files]
Ls={n:np.load(f) for n,f in zip(names,files)}   # (seed, 3, nrows) -> index 2 is L
NS=next(iter(Ls.values())).shape[0]
print(f'=== {FAM}: {len(names)} configs, {NS} seeds, {len(counts)} pairs')
R15=ap(-c.r.values); TAB=lg(c.tab.values)
res={'R15':float(R15.mean()),'tabicl_alone':float(ap(TAB).mean())}
per={}   # name -> list of per-pair series (one per seed) for the best deployed form
def ev_name(nm, Lseeds, betas=(0.5,1.0,2.0)):
    alone=[];stk={b:[] for b in betas}
    for si in range(NS):
        v=to_c(Lseeds[si]); alone.append(ap(v))
        for b in betas: stk[b].append(ap(TAB+b*lg(v)))
    out={f'{nm}|alone':float(np.mean([a.mean() for a in alone]))}
    for b in betas: out[f'{nm}|stack{b}']=float(np.mean([a.mean() for a in stk[b]]))
    per[f'{nm}|alone']=alone
    for b in betas: per[f'{nm}|stack{b}']=stk[b]
    return out
for n in names: res.update(ev_name(n,[Ls[n][si][2] for si in range(NS)]))
# ---- parameter-free fusions over configs ----
def fuse_logit(sub): return [np.mean([lg(Ls[n][si][2]) for n in sub],axis=0) for si in range(NS)]
def fuse_rank(sub):
    out=[]
    for si in range(NS):
        acc=np.zeros(len(s))
        for n in sub:
            acc+=pd.Series(Ls[n][si][2]).groupby(s.slot.values).rank(pct=True).values
        out.append(acc/len(sub))
    return out
ALL=names
groups={'ALL':ALL,'noview':[n for n in ALL if n not in ('lgb_kern','lgb_gplay','lgb_role')],'views':[n for n in ALL if n in ('lgb_kern','lgb_gplay','lgb_role','lgb_sym')]}
for gname,sub in groups.items():
    if len(sub)<2: continue
    fl=fuse_logit(sub)
    for si in range(NS): pass
    res.update({k.replace('X',''):v for k,v in ev_name(f'fuseLOGIT_{gname}',[1/(1+np.exp(-x)) for x in fl]).items()})
    fr=fuse_rank(sub)
    # rank fusion is not a probability: score directly and blend with tab by rank
    a=[];st=[]
    for si in range(NS):
        v=to_c(fr[si]); a.append(ap(v))
        rr=pd.Series(v).groupby(c.slot.values).rank(pct=True).values; tr=pd.Series(TAB).groupby(c.slot.values).rank(pct=True).values
        st.append(ap(0.5*rr+0.5*tr))
    res[f'fuseRANK_{gname}|alone']=float(np.mean([x.mean() for x in a])); per[f'fuseRANK_{gname}|alone']=a
    res[f'fuseRANK_{gname}|rank50tab']=float(np.mean([x.mean() for x in st])); per[f'fuseRANK_{gname}|rank50tab']=st
top=sorted(res.items(),key=lambda kv:-kv[1])
print(json.dumps({k:round(v,4) for k,v in top[:28]},indent=1))
json.dump({k:round(v,5) for k,v in res.items()},open(f'{R5}/e2_fuse_{ab}.json','w'),indent=1)
# bootstrap best vs reference
REF=os.environ.get('REF','lgb_sym|stack1.0')
if REF in per:
    best=[k for k,_ in top if k in per and k!=REF][0]
    rng=np.random.default_rng(7); idx=np.array(counts.index)
    d=np.mean([per[best][si].reindex(idx).values-per[REF][si].reindex(idx).values for si in range(NS)],axis=0)
    bs=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(4000)])
    print(f'best={best} {res[best]:.4f}  ref={REF} {res[REF]:.4f}  delta={d.mean():+.4f}  P(>0)={float((bs>0).mean()):.3f}  wins={int((d>1e-12).sum())} losses={int((d<-1e-12).sum())}')
