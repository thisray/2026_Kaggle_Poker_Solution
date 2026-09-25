"""Small exact slate posterior + expected AP@5 decoder on a fixed top-12 pool.
Frozen-upstream exploration ONLY. Never supplies truth cardinality at inference.
No tuned grid: l2=10, one feature recipe, fixed pool folds. Both negative and
positive results must be retained. Requires numpy, pandas, scipy.
"""
from pathlib import Path
import argparse,itertools,json,time
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp


def state_masks(k,max_count=5):
 masks=[]
 for n in range(max_count+1):
  for c in itertools.combinations(range(k),n):
   m=np.zeros(k);m[list(c)]=1;masks.append(m)
 return np.asarray(masks)

def expected_ap_decode(p:np.ndarray,masks:np.ndarray,invdenom:np.ndarray,kout=5):
 """Exact Bayes prefix-set DP; p is a posterior on candidate relevance subsets."""
 k=masks.shape[1];w=p*invdenom[masks.sum(1).astype(int)]
 joint=(masks.T*w)@masks
 a=np.diag(joint).copy()
 # Diagonal is not a joint hit with a previously selected candidate.
 np.fill_diagonal(joint,0)
 dp={0:(0.,())}
 for n in range(1,kout+1):
  nxt={}
  for subset in itertools.combinations(range(k),n):
   mask=sum(1<<i for i in subset);best=(-np.inf,())
   for j in subset:
    prev=mask^(1<<j)
    val,seq=dp[prev]
    gain=(a[j]+sum(joint[i,j] for i in subset if i!=j))/n
    if val+gain>best[0]:best=(float(val+gain),seq+(j,))
   nxt[mask]=best
  dp=nxt
 return max(dp.values(),key=lambda x:x[0])

def ap(y,m):
 return float((y*np.cumsum(y)/np.arange(1,len(y)+1)).sum()/min(m,5))

def run(csv,out):
 st=time.time();out=Path(out);out.mkdir(parents=True,exist_ok=True)
 d=pd.read_csv(csv).sort_values(['slot','rank_u_r5b']).reset_index(drop=True)
 cols=[c for c in d if c.startswith(('z_','ev_','wit_','o_','DS_','S_','Q_','tpl_'))]
 mask=state_masks(12);ns=mask.sum(1).astype(int)
 countfeatures=np.eye(6)[ns][:,1:]
 ab=list(itertools.combinations(range(12),2));ii=np.array([a for a,b in ab]);jj=np.array([b for a,b in ab])
 pairmask=mask[:,ii]*mask[:,jj]
 # Normalize consistency by number of pairs, so unary log odds still controls count.
 pairmask/=np.maximum(1,ns*(ns-1)/2)[:,None]
 base=[];results=[];fstats=[]
 for f in sorted(d.fold.unique()):
  tr=d.fold!=f
  x=d[cols].to_numpy(float)
  mu=np.nanmean(x[tr],0);sd=np.nanstd(x[tr],0)+1e-9
  x=np.nan_to_num(np.clip((x-mu)/sd,-6,6))
  features=[];truth=[];groups=[]
  for pid,g in d.groupby('slot',sort=False):
   g=g.head(12);idx=g.index.to_numpy();z=g.u_r5b.to_numpy()
   dist=-np.mean((x[idx[ii]]-x[idx[jj]])**2,1)
   cos=np.sum(x[idx[ii]]*x[idx[jj]],1)/(np.linalg.norm(x[idx[ii]],axis=1)*np.linalg.norm(x[idx[jj]],axis=1)+1e-9)
   score=np.c_[mask@z,countfeatures,pairmask@dist,pairmask@cos]
   y=g.ev.to_numpy(int)
   match=np.where((mask==y).all(1))[0]
   if len(match)!=1:raise ValueError('Need observed candidate label count <=5')
   features.append(score);truth.append(match[0]);groups.append(g)
  X=np.asarray(features);y=np.asarray(truth);train=np.array([g.fold.iloc[0]!=f for g in groups])
  xt=X[train];yt=y[train];gt=np.array([xt[i,j] for i,j in enumerate(yt)])
  n=len(xt);prior=np.array([1.,0,0,0,0,0,0,0])
  def fun(v):
   s=xt@v;p=np.exp(s-logsumexp(s,axis=1,keepdims=True))
   diff=v-prior
   loss=np.mean(logsumexp(s,axis=1)-s[np.arange(n),yt])+5.*np.sum(diff**2)/n
   grad=np.mean(np.einsum('ij,ijk->ik',p,xt)-gt,axis=0)+10*diff/n
   return loss,grad
  fit=minimize(fun,prior,jac=True,method='L-BFGS-B',bounds=[(.05,5)]+[(-20,20)]*7,options={'maxiter':100,'ftol':1e-9})
  # Learn denominator calibration only from outer-training labels, then freeze.
  inv=np.full(6,1/5.)
  counts_train=np.array([len(g[g.ev==1]) for g,t in zip(groups,train) if t])
  fullm=np.array([g.m_p.iloc[0] for g,t in zip(groups,train) if t])
  for m in range(6):
   vals=1/np.minimum(fullm[counts_train==m],5)
   inv[m]=(vals.sum()+5/5)/(len(vals)+5)
  fstats.append({'fold':int(f),'converged':bool(fit.success),'parameters':fit.x.tolist(),'inverse_denominator':inv.tolist()})
  for i,(g,t) in enumerate(zip(groups,train)):
   if t:continue
   s=X[i]@fit.x;p=np.exp(s-logsumexp(s))
   _,seq=expected_ap_decode(p,mask,inv)
   marg=p@mask;unary=np.argsort(-marg,kind='stable')[:5]
   y0=g.ev.to_numpy();m=int(g.m_p.iloc[0]);baseE=ap(y0[:5],m)
   results.append({'slot':int(g.slot.iloc[0]),'pool':int(g.pool.iloc[0]),'fold':int(f),'family':g.family.iloc[0],
    'base_E':baseE,'marginal_E':ap(y0[unary],m),'slate_E':ap(y0[list(seq)],m),
    'slate_order_original_ranks':','.join(str(j+1) for j in seq)})
  print('fold',f,'done',time.time()-st,flush=True)
 r=pd.DataFrame(results);r.to_csv(out/'per_pair.csv',index=False)
 summary={'E_base':float(r.base_E.mean()),'E_posterior_marginal':float(r.marginal_E.mean()),'E_expectedAP_slate':float(r.slate_E.mean()),
 'delta_slate':float((r.slate_E-r.base_E).mean()),'folds':fstats,'seconds':time.time()-st,
 'scope':'Fixed upstream r5b. Not validated against unavailable r7/LambdaRank per-row predictions. No deployment or LB.'}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('out');a=p.parse_args();run(a.csv,a.out)
