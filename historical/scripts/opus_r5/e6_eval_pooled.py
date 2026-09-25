"""R5-E6: score the pooled cross-family combiner on the CORRECT protocol
(full-truth denominators, identical to R4-G4 / R5-e3) and against the forms that are
actually deployed, not against an internal baseline."""
import numpy as np, pandas as pd, sys, os, json, glob
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18')
import ci_censored_event as C
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; R5=f'{O}/r5'
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9))); sg=lambda x:1/(1+np.exp(-x))
FAMS={'directed_transfer':'di','soft_play':'so','coordinated_isolation':'co'}
P=pd.read_parquet(f'{R5}/e5_pooled_oof.parquet')
cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
hmap=pd.read_parquet(f'{O}/np/hand_index.parquet').set_index('hand_id').hi
t45=pd.read_parquet(f'{O}/r3/t45_known_e_rerank.parquet'); t45['h']=t45.hand_id.map(hmap)
OOF=[c for c in P.columns if c.startswith('oof_')]
res={}
for fam,ab in FAMS.items():
    s=pd.read_parquet(f'{R5}/L/rows_{ab}.parquet').reset_index(drop=True); counts=s.groupby('slot').ev.sum()
    c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev']],on=['slot','h'],validate='one_to_one').merge(t45[['slot','h','tab']],on=['slot','h'],how='left')
    c=c.merge(P[P.fam==fam][['slot','h']+OOF],on=['slot','h'],validate='one_to_one').reset_index(drop=True)
    fmi=pd.MultiIndex.from_arrays([s.slot,s.h]); mi=pd.MultiIndex.from_arrays([c.slot,c.h]); to_c=lambda v: pd.Series(v,index=fmi).reindex(mi).values
    files=sorted(glob.glob(f'{R5}/L/{ab}__*.npy')); nms=[os.path.basename(f).split('__')[1][:-4] for f in files]
    Z={n:np.load(f) for n,f in zip(nms,files)}; NS=Z[nms[0]].shape[0]
    from g4_decode import decode
    pa=sg(np.mean([lg(Z[n][si][0]) for n in nms for si in range(NS)],axis=0)); pb=sg(np.mean([lg(Z[n][si][1]) for n in nms for si in range(NS)],axis=0))
    Lp=to_c(decode(s,np.clip(pa,1e-9,1-1e-6),np.clip(pb,1e-9,1-1e-6))[0])
    Lsym=to_c(np.mean([Z['lgb_sym'][si][2] for si in range(NS)],axis=0)); TAB=lg(c.tab.values); R15=-c.r.values
    rk=lambda v: pd.Series(v).groupby(c.slot.values).rank(pct=True).values
    DEP={'directed_transfer':TAB+1.0*lg(Lsym),'soft_play':R15.astype(float),'coordinated_isolation':TAB+3.0*lg(Lp)}[fam]
    out={'R15':float(C.pair_ap(c,R15,counts).mean()),'PFUSE_alone':float(C.pair_ap(c,Lp,counts).mean()),
         'DEPLOYED_form':float(C.pair_ap(c,DEP,counts).mean()),
         'SP_PFUSE_rank0.35':float(C.pair_ap(c,0.65*rk(R15)+0.35*rk(Lp),counts).mean())}
    for o in OOF:
        out[o]=float(C.pair_ap(c,c[o].values,counts).mean())
        for w in (0.3,0.5,0.7): out[f'{o}+dep{w}']=float(C.pair_ap(c,w*rk(c[o].values)+(1-w)*rk(DEP),counts).mean())
    res[fam]={k:round(v,4) for k,v in out.items()}
    top=sorted(res[fam].items(),key=lambda kv:-kv[1])[:8]
    print(f'{fam}: DEPLOYED {out["DEPLOYED_form"]:.4f} | top: {top}',flush=True)
json.dump(res,open(f'{R5}/e6_eval_pooled.json','w'),indent=1)
