"""Reproduce frozen-upstream evidence metrics and separate recoverable errors.
No training, no Kaggle submission, no use of IDs as predictors.
"""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd

SCORES=['u_r5b','z_cat','z_lr','z_lr11','rs_cat','rs_ranker','rs_blend','re_blend']
def ap5(y,m):
    y=np.asarray(y,float)[:5]
    return float(np.sum(y*np.cumsum(y)/np.arange(1,len(y)+1))/min(m,5))
def run(inp,out):
 d=pd.read_csv(inp);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 assert not d.duplicated(['slot','hand_id']).any()
 assert d.groupby('pool').fold.nunique().max()==1
 rows=[]
 for slot,g in d.groupby('slot',sort=False):
  y=g.ev.to_numpy(int);m=int(g.m_p.iloc[0]);assert (g.m_p==m).all()
  r={'slot':int(slot),'pool':int(g.pool.iloc[0]),'fold':int(g.fold.iloc[0]),'m_p':m,'candidate_positive':int(y.sum()),'candidate_n':len(g)}
  for c in SCORES:
   order=np.argsort(-g[c].to_numpy(),kind='stable')
   r[c]=ap5(y[order],m)
  order=np.argsort(-g.rs_blend.to_numpy(),kind='stable');yy=y[order]
  for k in [1,2,3,5,8,12,20]:
   r[f'hits{k}']=int(yy[:k].sum());r[f'recall{k}']=float(yy[:k].sum()/m)
   r[f'oracle{k}']=float(min(yy[:k].sum(),5)/min(m,5))
  r['selected5_order_loss']=r['oracle5']-r['rs_blend']
  r['top12_selection_loss']=r['oracle12']-r['oracle5']
  r['rank13_20_loss']=r['oracle20']-r['oracle12']
  r['outside20_loss']=1-r['oracle20']
  r['best_model_oracle']=max(r[c] for c in SCORES)
  # Keep best observed rank from each model only for oracle coverage (truth used).
  unions={}
  for k in [5,8]:
   ix=np.unique(np.concatenate([np.argsort(-g[c].to_numpy(),kind='stable')[:k] for c in SCORES]))
   r[f'union{k}_oracle']=min(int(y[ix].sum()),5)/min(m,5)
  rows.append(r)
 p=pd.DataFrame(rows);p.to_csv(out/'per_pair_oracles.csv',index=False)
 summ={'scope':'Frozen upstream, supplied 372 labelled positive pairs; not a new trained model or LB result',
       'rows':len(d),'pairs':len(p),'pools':d.pool.nunique(),'truth_hands':int(p.m_p.sum()),'retrieved_truth':int(d.ev.sum()),
       'mean':p.select_dtypes('number').drop(columns=['slot','pool','fold']).mean().to_dict(),
       'folds':p.groupby('fold')[['rs_blend','u_r5b','selected5_order_loss','top12_selection_loss','outside20_loss']].mean().to_dict('index'),
       'ties_top5_pairs':sum(bool(g.rs_blend.nlargest(6).duplicated().any()) for _,g in d.groupby('slot')),
       'pair_damage':{c:{'win':int((p[c]>p.rs_blend+1e-12).sum()),'loss':int((p[c]<p.rs_blend-1e-12).sum()),'tie':int((abs(p[c]-p.rs_blend)<1e-12).sum())} for c in SCORES}}
 # Bootstrap pools, preserving clustered pairs. Descriptive uncertainty only.
 rng=np.random.default_rng(20260918);pool=p.groupby('pool').agg(n=('slot','size'),s=('rs_blend','sum'))
 ix=rng.integers(0,len(pool),(5000,len(pool)));boot=pool.s.to_numpy()[ix].sum(1)/pool.n.to_numpy()[ix].sum(1)
 summ['baseline_pool_bootstrap_CI95']=np.quantile(boot,[.025,.975]).tolist()
 (out/'oracle_summary.json').write_text(json.dumps(summ,indent=2))
 print(json.dumps(summ,indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--input',required=True);a.add_argument('--out',required=True);x=a.parse_args();run(x.input,x.out)
