"""R5-E8: R5-e7 showed the per-pair choice of event-model config is STABLE across
seeds (DT 0.810 held-out vs 0.743 for the best single config), so 'which model is
right here' is a real pair-level property, not noise. Can it be predicted from
PAIR-LEVEL features alone (no labels at prediction time)?
  (a) supervised selector: regress each config's per-pair AP on pair x config
      features, pool GroupKFold, then take the argmax / top-k average per pair;
  (b) unsupervised consensus weighting: down-weight configs whose within-pair
      ranking is far from the median ranking (parameter-free).
Trained on two seeds, scored on the held-out seed, rotated."""
import numpy as np, pandas as pd, sys, os, json, glob, lightgbm as lgb
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18')
import ci_censored_event as C
from sklearn.model_selection import GroupKFold
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; R5=f'{O}/r5'
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9)))
cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
hmap=pd.read_parquet(f'{O}/np/hand_index.parquet').set_index('hand_id').hi
t45=pd.read_parquet(f'{O}/r3/t45_known_e_rerank.parquet'); t45['h']=t45.hand_id.map(hmap)
res={}
for fam,ab in (('directed_transfer','di'),('soft_play','so'),('coordinated_isolation','co')):
    s=pd.read_parquet(f'{R5}/L/rows_{ab}.parquet').reset_index(drop=True); counts=s.groupby('slot').ev.sum()
    c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev']],on=['slot','h'],validate='one_to_one').merge(t45[['slot','h','tab']],on=['slot','h'],how='left').reset_index(drop=True)
    fmi=pd.MultiIndex.from_arrays([s.slot,s.h]); mi=pd.MultiIndex.from_arrays([c.slot,c.h]); to_c=lambda v: pd.Series(v,index=fmi).reindex(mi).values
    files=sorted(glob.glob(f'{R5}/L/{ab}__*.npy')); nms=[os.path.basename(f).split('__')[1][:-4] for f in files]
    Z={n:np.load(f) for n,f in zip(nms,files)}; NS=Z[nms[0]].shape[0]
    idx=np.array(counts.index); pool=idx//900
    Lc={(n,si):to_c(Z[n][si][2]) for n in nms for si in range(NS)}
    AP={(n,si):C.pair_ap(c,Lc[(n,si)],counts).reindex(idx).values for n in nms for si in range(NS)}
    nh=s.groupby('slot').size().reindex(idx).values
    rows=[]
    for si in range(NS):
        R={n:pd.Series(Lc[(n,si)]).groupby(c.slot.values).rank(pct=True) for n in nms}
        med=pd.concat(R.values(),axis=1).median(axis=1)
        sA={n:pd.Series(Z[n][si][0]).groupby(s.slot.values).sum().reindex(idx).values for n in nms}
        sB={n:pd.Series(Z[n][si][1]).groupby(s.slot.values).sum().reindex(idx).values for n in nms}
        top5={n:pd.Series(Lc[(n,si)]).groupby(c.slot.values).apply(lambda v: np.sort(v)[-5:].sum()).reindex(idx).values for n in nms}
        agree={n:R[n].groupby(c.slot.values).apply(lambda v: np.corrcoef(v, med.loc[v.index])[0,1]).reindex(idx).values for n in nms}
        for ci,n in enumerate(nms):
            rows.append(pd.DataFrame(dict(slot=idx,pool=pool,seed=si,cfg=ci,y=AP[(n,si)],nh=nh,
                sumA=sA[n],sumB=sB[n],top5=top5[n],agree=agree[n],
                sumA_rel=sA[n]/np.mean([sA[m] for m in nms],axis=0),sumB_rel=sB[n]/np.clip(np.mean([sB[m] for m in nms],axis=0),1e-6,None),
                spreadA=np.std([sA[m] for m in nms],axis=0),spreadB=np.std([sB[m] for m in nms],axis=0),
                agree_mean=np.mean([agree[m] for m in nms],axis=0))))
    D=pd.concat(rows,ignore_index=True)
    FE=['cfg','nh','sumA','sumB','top5','agree','sumA_rel','sumB_rel','spreadA','spreadB','agree_mean']
    pools=np.array(sorted(np.unique(pool)))
    sel_sc=[];cons_sc=[];best_sc=[];mean_sc=[]
    for te in range(NS):
        tr=[x for x in range(NS) if x!=te]
        Dtr=D[D.seed.isin(tr)]; Dte=D[D.seed==te]
        pred=np.zeros(len(Dte))
        for _,vi in GroupKFold(5).split(pools,groups=pools):
            vp=set(pools[vi]); m_tr=~Dtr.pool.isin(vp); m_te=Dte.pool.isin(vp).values
            mdl=lgb.LGBMRegressor(n_estimators=300,learning_rate=.05,num_leaves=15,min_child_samples=40,colsample_bytree=.8,reg_lambda=10,verbosity=-1,n_jobs=8,random_state=7).fit(Dtr.loc[m_tr,FE],Dtr.loc[m_tr,'y'])
            pred[m_te]=mdl.predict(Dte.loc[m_te,FE])
        Dte=Dte.assign(pred=pred)
        pick=Dte.loc[Dte.groupby('slot').pred.idxmax()].set_index('slot').cfg
        sel_sc.append(float(np.mean([AP[(nms[pick[p]],te)][i] for i,p in enumerate(idx)])))
        # (b) unsupervised consensus weighting: weight = softmax(agree * 8) per pair, fuse logit L
        W=Dte.pivot_table(index='slot',columns='cfg',values='agree').reindex(idx).values
        W=np.exp(8*(W-np.nanmax(W,axis=1,keepdims=True))); W=np.nan_to_num(W); W/=W.sum(1,keepdims=True)
        wmap={p:W[i] for i,p in enumerate(idx)}
        Lstack=np.stack([lg(Lc[(n,te)]) for n in nms])           # (cfg, rows)
        wrow=np.stack([wmap[sl] for sl in c.slot.values])         # (rows, cfg)
        fused=(Lstack.T*wrow).sum(1)
        cons_sc.append(float(C.pair_ap(c,fused,counts).mean()))
        trm=pd.DataFrame({n:np.mean([AP[(n,x)] for x in tr],axis=0) for n in nms}).mean()
        best_sc.append(float(np.mean(AP[(trm.idxmax(),te)]))); mean_sc.append(float(np.mean([np.mean(AP[(n,te)]) for n in nms])))
    res[fam]=dict(mean_of_configs=round(float(np.mean(mean_sc)),4),single_best_config=round(float(np.mean(best_sc)),4),
                  supervised_selector=round(float(np.mean(sel_sc)),4),consensus_weighted_fusion=round(float(np.mean(cons_sc)),4))
    print(fam,res[fam],flush=True)
json.dump(res,open(f'{R5}/e8_selector.json','w'),indent=1)
