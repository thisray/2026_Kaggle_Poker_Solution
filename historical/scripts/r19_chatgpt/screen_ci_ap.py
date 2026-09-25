import os,sys
from pathlib import Path
WORK=Path(os.environ.get("R19_WORK","./r19_work"));WORK.mkdir(parents=True,exist_ok=True)
R18_CODE=Path(os.environ["R18_CODE"])
R19_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R18_CODE));sys.path.insert(0,str(R19_ROOT/"code"))
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.isotonic import IsotonicRegression
from capped_ap import moments,optimize
import ci_censored_event as C
W=WORK; O=R19_ROOT/'results'
ALL=[];details=[]
for seed in [260919,11,29,47]:
 s=pd.read_pickle(W/f'ci_full_{seed}.pkl');c=pd.read_pickle(W/f'ci_cand_{seed}.pkl')
 c=c.merge(s[['slot','h','fold']],on=['slot','h'],validate='1:1')
 bprob=np.zeros(len(c))
 for f in range(5):
  va=c.fold==f;tr=~va
  iso=IsotonicRegression(increasing=False,out_of_bounds='clip').fit(c.loc[tr,'r'],c.loc[tr,'ev'])
  bprob[va]=iso.predict(c.loc[va,'r'])
 c['br_prob']=bprob
 scores={n:[] for n in ['R18hard','event_marginal','event_exact','mixture_marginal','mixture_exact','mixture_exact_hard']}
 for sl,g in s.sort_values(['slot','ts','h'],kind='stable').groupby('slot',sort=False):
  g=g.reset_index(drop=True);j=c[c.slot==sl].copy();j['pos']=j.h.map(dict(zip(g.h,range(len(g)))))
  # Fixed 15-candidate screen: union of R18 top10 and event top10, then priority union.
  use=list(dict.fromkeys(j.sort_values('newscore',ascending=False).h.head(10).tolist()+j.sort_values('q',ascending=False).h.head(10).tolist()))[:15]
  j=j[j.h.isin(use)].sort_values('pos').reset_index(drop=True)
  a,b=moments(g.p.to_numpy(),j.pos.to_numpy(),constant_denominator=True)
  teach=j.br_prob.to_numpy();aa=.5*a+.5*teach/5;bb=.5*b+.5*np.outer(teach,teach)/5;np.fill_diagonal(bb,aa)
  ev,evval=optimize(a,b);mix,mval=optimize(aa,bb)
  hard=(~((j.pa_at_trig==6)&j.y1.isin([2,3]))).to_numpy()
  ah=aa.copy();bh=bb.copy();ah[hard]-=100
  hm,_=optimize(ah,bh)
  base=c[c.slot==sl].copy();base['score']=base.newscore-100*(~((base.pa_at_trig==6)&base.y1.isin([2,3])))
  orders={'R18hard':base.sort_values(['score','ts','h'],ascending=[False,True,True]).h.head(5).tolist(),
   'event_marginal':j.iloc[np.argsort(-a,kind='stable')[:5]].h.tolist(),'event_exact':j.iloc[ev].h.tolist(),
   'mixture_marginal':j.iloc[np.argsort(-aa,kind='stable')[:5]].h.tolist(),'mixture_exact':j.iloc[mix].h.tolist(),'mixture_exact_hard':j.iloc[hm].h.tolist()}
  truth=set(g[g.ev].h);res={'seed':seed,'slot':int(sl),'fold':int(g.fold.iloc[0])}
  for name,order in orders.items():
   y=np.array([h in truth for h in order]);ap=float((y*np.cumsum(y)/np.arange(1,6)).sum()/len(truth));scores[name].append(ap);res[name]=ap
  details.append(res)
 row={'seed':seed,**{k:float(np.mean(v)) for k,v in scores.items()}}
 ALL.append(row);print(json.dumps(row),flush=True)
O.joinpath('ci_exact_ap_screen.json').write_text(json.dumps(ALL,indent=2));pd.DataFrame(details).to_csv(O/'ci_exact_ap_pairs.csv',index=False)
