import os,sys
from pathlib import Path
WORK=Path(os.environ.get("R19_WORK","./r19_work"));WORK.mkdir(parents=True,exist_ok=True)
R18_CODE=Path(os.environ["R18_CODE"])
R19_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R18_CODE));sys.path.insert(0,str(R19_ROOT/"code"))
from pathlib import Path
import sys,json,time
import pandas as pd,numpy as np
import ci_censored_event as C
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
W=WORK
a=C.prepare(pd.read_pickle(W/'t5_dev_seq.pkl'));c=pd.read_pickle(W/'t4_wrong_vs_hit.pkl')
pools=np.sort(a.pool.unique());s=a[a.fam==C.FAMILY].reset_index(drop=True);c=c[c.slot.isin(s.slot)].copy()
c=c.drop(columns=['ts','ev','y1','pa_at_trig'],errors='ignore').merge(s[['slot','h','ts','ev','y1','pa_at_trig']],on=['slot','h'],validate='1:1')
mask=C.uncensored_training_rows(s)
for seed in [260919,11,29,47]:
 p=np.zeros(len(s));folds=np.full(len(s),-1)
 for f,(_,i) in enumerate(GroupKFold(5,shuffle=True,random_state=seed).split(pools,groups=pools)):
  va=s.pool.isin(pools[i]).to_numpy();tr=(~va)&mask
  m=lgb.LGBMClassifier(**C.PARAMS);m.fit(s.loc[tr,C.FEATURES].astype(float),s.loc[tr,'ev']);p[va]=m.predict_proba(s.loc[va,C.FEATURES].astype(float))[:,1];folds[va]=f
 q=C.first_k_marginal(s,p);j=C.rank_candidates(s,c,q)
 counts=s.groupby('slot').ev.sum();b=C.pair_ap(j,-j.r,counts);r3=C.pair_ap(j,-j.r-100*(~((j.pa_at_trig==6)&j.y1.isin([2,3]))),counts)
 r18=C.pair_ap(j,j.newscore,counts);hard=C.pair_ap(j,j.newscore-100*(~((j.pa_at_trig==6)&j.y1.isin([2,3]))),counts)
 s.assign(p=p,q=q,fold=folds).to_pickle(W/f'ci_full_{seed}.pkl');j.to_pickle(W/f'ci_cand_{seed}.pkl')
 print(json.dumps({'seed':seed,'R15':b.mean(),'R3':r3.mean(),'R18':r18.mean(),'R18hard':hard.mean()}),flush=True)
