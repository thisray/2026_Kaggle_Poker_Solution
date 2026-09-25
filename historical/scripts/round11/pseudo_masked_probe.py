"""Mask HUMAN evidence labels on half the outer-training pools.
Compares real-label acquisition with self-labeling: no evaluation labels are used
for pseudo-labels or training. The supplied candidate pool/upstream features are
frozen and may encode prior training, so this is NOT an independent label-scaling
or production G1 estimate. No legacy prediction, true family or cardinality inputs.
"""
from pathlib import Path
import argparse,json,time
import numpy as np,pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from scipy.special import logit

def fit(x,y,w,seed=1101):
 return lgb.train(dict(objective='cross_entropy',learning_rate=.05,num_leaves=15,min_data_in_leaf=40,
 feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,lambda_l2=10,verbosity=-1,num_threads=2,
 seed=seed,deterministic=True,force_col_wise=True),lgb.Dataset(x,label=y,weight=w),num_boost_round=200)
def ap(y,m):
 y=y[:5];return float((y*np.cumsum(y)/np.arange(1,len(y)+1)).sum()/min(m,5))
def run(csv,out):
 start=time.time();out=Path(out);out.mkdir(parents=True,exist_ok=True)
 d=pd.read_csv(csv).sort_values(['slot','hand_id']).reset_index(drop=True)
 # Current decision/gameplay fields, no ranks, prior evidence predictions or IDs.
 cols=[c for c in d if (c.startswith(('DS_','S_','wit_','ev_')) or c in
 ['eq_fold_to_mx','facing_mx','eq_aggr_active_mx','P_pos_mx','P_eq_first_mn','flow_mx'])]
 x=np.nan_to_num(d[cols].to_numpy(float),nan=0,posinf=30,neginf=-30).clip(-30,30)
 y=d.ev.to_numpy(float);m=d.m_p.to_numpy(float);pool=d.pool.to_numpy();slot=d.slot.to_numpy()
 result=d[['slot','pool','hand_id','fold','ev','m_p']].copy();stats=[]
 arms=['human_half','hard_top5','soft_all','selective_soft','human_full_oracle']
 for arm in arms:result[arm]=np.nan
 for f in sorted(d.fold.unique()):
  outer=d.fold.to_numpy()!=f;val=~outer
  pools=np.unique(pool[outer]);rng=np.random.default_rng(1101+int(f));rng.shuffle(pools)
  labelled=np.isin(pool,pools[:len(pools)//2])&outer;hidden=outer&~labelled
  innerpred=np.full(len(d),np.nan)
  lp=np.unique(pool[labelled]);rng.shuffle(lp)
  for g in range(3):
   iv=labelled&np.isin(pool,lp[g::3]);it=labelled&~iv
   model=fit(x[it],y[it],1/m[it]);innerpred[iv]=model.predict(x[iv])
  cal=LogisticRegression(C=1,max_iter=500).fit(logit(np.clip(innerpred[labelled],1e-5,1-1e-5))[:,None],y[labelled])
  teacher=fit(x[labelled],y[labelled],1/m[labelled])
  raw=teacher.predict(x[hidden]);q=cal.predict_proba(logit(np.clip(raw,1e-5,1-1e-5))[:,None])[:,1]
  h=d.loc[hidden,['slot']].copy();h['q']=q
  rank=h.groupby('slot').q.rank(ascending=False,method='first').to_numpy();hard=(rank<=5).astype(float)
  keep=(q>=.85)|(q<=.05)
  # A pseudo query has TOTAL weight <= 1; human pair total is 20/m=4..6.67.
  ww=np.full(len(h),1/20.)
  result.loc[val,'human_half']=teacher.predict(x[val])
  for arm,targets,weights in [('hard_top5',hard,ww),('soft_all',q,ww),('selective_soft',q,ww*keep),
                              ('human_full_oracle',y[hidden],1/m[hidden])]:
   allx=np.r_[x[labelled],x[hidden]];ally=np.r_[y[labelled],targets];allw=np.r_[1/m[labelled],weights]
   use=allw>0;model=fit(allx[use],ally[use],allw[use]);result.loc[val,arm]=model.predict(x[val])
  yh=y[hidden]
  stats.append({'fold':int(f),'labelled_pairs':int(d[labelled].slot.nunique()),'hidden_pairs':int(d[hidden].slot.nunique()),
   'hard_top5_precision':float(yh[hard==1].mean()),'selective_positive_count':int((q>=.85).sum()),
   'selective_positive_precision':float(yh[q>=.85].mean()) if (q>=.85).any() else None,
   'selective_negative_count':int((q<=.05).sum()),
   'selective_negative_precision':float((1-yh[q<=.05]).mean()) if (q<=.05).any() else None})
  print('fold',f,'completed',time.time()-start,flush=True)
 result.to_csv(out/'scores.csv.gz',index=False,compression='gzip')
 rows=[]
 for pid,g in result.groupby('slot'):
  row={'slot':int(pid),'pool':int(g.pool.iloc[0]),'fold':int(g.fold.iloc[0])}
  for arm in arms:
   yy=g.sort_values(arm,ascending=False,kind='stable').ev.to_numpy();row[arm]=ap(yy,int(g.m_p.iloc[0]))
  rows.append(row)
 r=pd.DataFrame(rows);r.to_csv(out/'per_pair.csv',index=False)
 summary={'E':{arm:float(r[arm].mean()) for arm in arms},'delta_vs_human_half':{arm:float((r[arm]-r.human_half).mean()) for arm in arms},
 'fold_deltas':{arm:{str(int(f)):float(v) for f,v in (r[arm]-r.human_half).groupby(r.fold).mean().items()} for arm in arms},
 'pseudo_audit':stats,'features':cols,'seconds':time.time()-start,
 'scope':'Masked-label proxy on frozen r5b-selected positive-pair candidates. Not a production G1 estimate or independent label scaling. Baseline is this proxy teacher, not r7. Pair positivity is known in this simulation; unknown-pair precision is not tested.'}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('out');a=p.parse_args();run(a.csv,a.out)
