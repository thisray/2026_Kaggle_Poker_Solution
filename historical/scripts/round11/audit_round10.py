"""Recompute identifiable Round10 diagnostics from the supplied narrow table.
Does not claim to reproduce unavailable moments models or hidden-pair predictions.
"""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from scipy.stats import ttest_1samp

def ap(y,m):
 r=np.asarray(y)[:5];return float((r*np.cumsum(r)/np.arange(1,len(r)+1)).sum()/min(m,5))
def run(path,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 n=pd.read_csv(path).sort_values(['slot','rank_u_r5b']).reset_index(drop=True)
 fs=[c for c in n if c.startswith(('z_','ev_','wit_','o_','DS_','S_','Q_','tpl_'))]
 x=n[fs].to_numpy(float);x=np.clip((x-np.nanmean(x,0))/(np.nanstd(x,0)+1e-9),-6,6)
 top=n[n.rank_u_r5b<=12];xt=x[top.index];records=[]
 for slot,g in top.groupby('slot',sort=False):
  idx=g.index.to_numpy(); yy=g.ev.to_numpy(bool);a,b=np.triu_indices(len(g),1)
  ee=yy[a]&yy[b];ef=yy[a]^yy[b]
  if not ee.any() or not ef.any():continue
  # Literal original uses xt[0:len(g)] for EVERY query.
  bug=-np.mean((xt[a]-xt[b])**2,1)
  good=-np.mean((x[idx[a]]-x[idx[b]])**2,1)
  records.append({'slot':int(slot),'pool':int(g.pool.iloc[0]),'family':g.family.iloc[0],
   'bug_diff':float(bug[ee].mean()-bug[ef].mean()),'correct_diff':float(good[ee].mean()-good[ef].mean())})
 r=pd.DataFrame(records);r.to_csv(out/'consistency_by_pair.csv',index=False)
 s={}
 rng=np.random.default_rng(1101)
 for col in ['bug_diff','correct_diff']:
  v=r[col].to_numpy();s[col]={'mean':float(v.mean()),'positive_fraction':float((v>0).mean()),
    'legacy_t_like':float(v.mean()/(v.std()/np.sqrt(len(v))+1e-9)),
    'paired_t':float(ttest_1samp(v,0).statistic)}
  sums=r.groupby('pool')[col].sum().to_numpy();counts=r.groupby('pool').size().to_numpy()
  ix=rng.integers(0,len(sums),(4000,len(sums)));boot=sums[ix].sum(1)/counts[ix].sum(1)
  s[col]['pool_bootstrap_95']=np.quantile(boot,[.025,.975]).tolist()
 group=[]
 for slot,g in n.groupby('slot',sort=False):
  yy=g.ev.to_numpy(int);m=int(g.m_p.iloc[0]);nr=int(yy.sum());t=g.ts_pct_in_pair.to_numpy()
  last=t[yy==1].max() if nr else -np.inf
  z=g.u_r5b.to_numpy();after=(yy==0)&(t>last)
  group.append({'slot':int(slot),'pool':int(g.pool.iloc[0]),'family':g.family.iloc[0],
   'm_p':m,'candidate_relevant':nr,'E_correct':ap(yy,m),'E_candidate_denominator':ap(yy,nr) if nr else 0,
   'late_non_evidence':int(after.sum()),'late_z_gt_point5':int((after&(z>.5)).sum()),
   'late_in_top12':int((after&(g.rank_u_r5b.to_numpy()<=12)).sum()),
   'u_max':float(z.max()),'hits5':int(yy[:5].sum())})
 q=pd.DataFrame(group);q.to_csv(out/'geometry_denominator_and_late.csv',index=False)
 ranks=n.groupby('rank_u_r5b').ev.agg(['count','sum','mean']);ranks.to_csv(out/'rank_precision.csv')
 pseudo=[]
 for k in [1,2,3,4,5]:
  pick=n[n.rank_u_r5b<=k]
  per=pick.groupby('slot').ev.sum()
  pseudo.append({'k':k,'selected':len(pick),'correct':int(pick.ev.sum()),'precision':float(pick.ev.mean()),
    'all_selected_correct_pair_fraction':float((per==k).mean())})
 s.update({'rows':len(n),'pairs':n.slot.nunique(),'pools':n.pool.nunique(),'features':len(fs),
 'E_r5b_correct':float(q.E_correct.mean()),'E_r5b_wrong_candidate_denom':float(q.E_candidate_denominator.mean()),
 'missing_evidence_pairs':int((q.m_p>q.candidate_relevant).sum()),
 'missing_evidence_hands':int((q.m_p-q.candidate_relevant).sum()),
 'late_non_evidence':int(q.late_non_evidence.sum()),'late_top12':int(q.late_in_top12.sum()),
 'late_score_gt_0_5':int(q.late_z_gt_point5.sum()),'u_r5b_quantile':np.quantile(n.u_r5b,[0,.25,.5,.75,.99,1]).tolist(),
 'pseudo_precision':pseudo,
 'limitations':['Fixed r5b upstream OOF; not r7 or LambdaRank outputs.',
 'Consistency truth comparison is diagnostic, not a deployable prediction.',
 'Pseudo precision measured only on known positive development pairs; unknown-pair precision not measured.',
 'Candidate-denominator bias evaluated on r5b, not unavailable x10 predictions.']})
 (out/'audit_summary.json').write_text(json.dumps(s,indent=2))
 print(json.dumps(s,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('out');a=p.parse_args();run(a.csv,a.out)
