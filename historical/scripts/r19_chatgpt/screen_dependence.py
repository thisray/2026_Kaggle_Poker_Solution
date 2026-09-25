"""Exploratory HMM time-shuffle control and initial Legendre random-rate screen.
Use f4_random_effects.py for final Gauss-Jacobi results.
"""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
from scipy.special import expit,betaln,logsumexp
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from numba import njit
B=Path(__file__).resolve().parents[1]; d=pd.read_csv(B/'results/f4_gate_input.csv.gz').sort_values(['slot','ts']).reset_index(drop=True)
@njit
def hmm_score(par,lf,slot):
 p01=1/(1+np.exp(-par[0]));p11=1/(1+np.exp(-par[1]));init=p01/(1-p11+p01);pr=init;total=0.
 for i in range(len(lf)):
  if i==0 or slot[i]!=slot[i-1]:pr=init
  z=np.log(pr)-np.log1p(-pr)+lf[i];total+=np.logaddexp(np.log1p(-pr),np.log(pr)+lf[i]);post=1/(1+np.exp(-z));pr=p01*(1-post)+p11*post
 return total

def hfit(x):
 best=None
 for st in [[.89,.89],[0.,1.5]]:
  r=minimize(lambda p:-hmm_score(p,x.lf.values,x.slot.values),st,method='L-BFGS-B',bounds=[(-6,6),(-6,6)])
  if best is None or r.fun<best.fun:best=r
 return best.x
# Gauss-Legendre on rho, normalized beta weights. Quad doubling checked below.
def table(x,n=80):
 rho,w=np.polynomial.legendre.leggauss(n);rho=(rho+1)/2;w=w/2
 ll=np.logaddexp(np.log1p(-rho)[None,:],np.log(rho)[None,:]+x.lf.values[:,None]);parts=[]
 for _,ids in x.groupby('slot',sort=False).indices.items():parts.append(ll[ids].sum(0))
 return rho,w,np.stack(parts)
def rscore(par,tab):
 rho,w,L=tab;mu=expit(par[0]);k=np.exp(par[1]);aa=mu*k;bb=(1-mu)*k
 lw=np.log(w)+(aa-1)*np.log(rho)+(bb-1)*np.log1p(-rho)-betaln(aa,bb);lw-=logsumexp(lw)
 return float(logsumexp(lw[None,:]+L,axis=1).sum())
def rfit(x):
 tab=table(x)
 rs=[minimize(lambda p:-rscore(p,tab),[.89,np.log(k)],method='L-BFGS-B',bounds=[(-3,3),(-1,8)]) for k in [3,30,300]]
 return min(rs,key=lambda r:r.fun).x
rows=[];rng=np.random.default_rng(190919)
for seed in [260919,11,29,47]:
 h=0.;r=0.;sh=np.zeros(20);folds=[]
 for fold,(tr,va) in enumerate(GroupKFold(5,shuffle=True,random_state=seed).split(d,groups=d.pool)):
  train=d.iloc[tr].reset_index(drop=True);valid=d.iloc[va].reset_index(drop=True);hp=hfit(train);rp=rfit(train)
  hv=hmm_score(hp,valid.lf.values,valid.slot.values);rv=rscore(rp,table(valid));h+=hv;r+=rv
  for j in range(len(sh)):
   lf=valid.lf.values.copy()
   for _,idx in valid.groupby('slot',sort=False).indices.items():lf[idx]=rng.permutation(lf[idx])
   sh[j]+=hmm_score(hp,lf,valid.slot.values)
  folds.append({'fold':fold,'HMM_llr':hv,'random_rho_llr':rv,'mu':float(expit(rp[0])),'concentration':float(np.exp(rp[1]))})
 old=next(i['heldout_LLR'] for i in json.loads((B/'results/f4_rho_gate_cv.json').read_text())['cv'] if i['seed']==seed and i['model']=='constant')
 row={'seed':seed,'constant_llr':old,'HMM_llr':h,'random_rho_llr':r,'random_rho_delta':r-old,'shuffled_HMM_llr_mean':float(sh.mean()),'shuffled_HMM_sd':float(sh.std()),'actual_minus_shuffle_mean':h-float(sh.mean()),'folds':folds};rows.append(row);print(json.dumps(row),flush=True)
p=rfit(d);full={'mu':float(expit(p[0])),'concentration':float(np.exp(p[1])),'llr80':rscore(p,table(d,80)),'llr160':rscore(p,table(d,160)),'params':p.tolist()}
(B/'results/f4_dependence_control.json').write_text(json.dumps({'status':'frozen-policy selected-member grouped held-out likelihood, no evidence labels','cv':rows,'full':full},indent=2));print(full,flush=True)
