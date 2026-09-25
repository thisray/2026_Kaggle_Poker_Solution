import os,sys
from pathlib import Path
WORK=Path(os.environ.get("R19_WORK","./r19_work"));WORK.mkdir(parents=True,exist_ok=True)
R18_CODE=Path(os.environ["R18_CODE"])
R19_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R18_CODE));sys.path.insert(0,str(R19_ROOT/"code"))
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
import ci_censored_event as C
from cardinality_prior import conditional_first_k
W=WORK;O=R19_ROOT/'results';rows=[];detail=[]
seeds=[260919,11,29,47]
for seed in seeds+['ensemble']:
 s=pd.read_pickle(W/f'ci_full_{seeds[0] if seed=="ensemble" else seed}.pkl');c=pd.read_pickle(W/f'ci_cand_{seeds[0] if seed=="ensemble" else seed}.pkl')
 if seed=='ensemble':s.p=np.mean([pd.read_pickle(W/f'ci_full_{ss}.pkl').p.values for ss in seeds],axis=0)
 q0=C.first_k_marginal(s,s.p);q1=np.zeros(len(s));probs=[]
 for sl,g in s.sort_values(['slot','ts','h']).groupby('slot',sort=False):
  v,z=conditional_first_k(g.p.to_numpy());q1[g.index]=v;probs.append(z)
 counts=s.groupby('slot').ev.sum();d={'seed':seed,'P_T_ge_5_min':float(min(probs)),'P_T_ge_5_mean':float(np.mean(probs))}
 for name,q in [('standard',q0),('ge5',q1)]:
  j=C.rank_candidates(s,c,q);a=C.pair_ap(j,j.newscore,counts);b=C.pair_ap(j,j.newscore-100*(~((j.pa_at_trig==6)&j.y1.isin([2,3]))),counts)
  d[name]=a.mean();d[name+'_hard']=b.mean()
  detail.extend([{'seed':seed,'model':name,'slot':int(sl),'ap':float(a.loc[sl]),'ap_hard':float(b.loc[sl])} for sl in counts.index])
 rows.append(d);print(json.dumps(d),flush=True)
O.joinpath('ci_cardinality_prior.json').write_text(json.dumps(rows,indent=2));pd.DataFrame(detail).to_csv(O/'ci_cardinality_pairs.csv',index=False)
