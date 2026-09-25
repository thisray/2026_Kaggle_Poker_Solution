"""R19 pair-specific latent activation rates, with uncertainty integrated in first-K decoding.

Input: CSV(.gz)/Parquet, slot,h,pool,lf,l0,ts; optional success (0/1) and
hand_id. lf/l0 are R3's frozen decision-emission sufficient statistics.
This is an unsupervised selected-member likelihood model, NOT evidence validation.
The CLI does not submit, promote pairs, or replace behavior labels.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logsumexp,roots_jacobi
from sklearn.model_selection import GroupKFold

def quadrature(mu:float,kappa:float,n:int=160):
    if not 0<mu<1 or kappa<=0: raise ValueError('Invalid Beta hyperparameters')
    a,b=mu*kappa,(1-mu)*kappa
    nodes,w=roots_jacobi(n,b-1,a-1)
    rho=(nodes+1)/2
    return rho,np.log(w)-logsumexp(np.log(w))

def validate(d:pd.DataFrame):
    required={'slot','h','pool','lf','l0','ts'}
    if not required.issubset(d): raise ValueError(f'Missing columns: {required-set(d)}')
    if d.duplicated(['slot','h']).any(): raise ValueError('Duplicate pair-hands')
    if not np.isfinite(d[['lf','l0','ts']].to_numpy()).all(): raise ValueError('Nonfinite emissions/time')
    if (d.l0>d.lf+1e-7).any(): raise ValueError('lf must include the no-activity mass l0')
    return d.sort_values(['slot','ts','h']).reset_index(drop=True)

def pair_tables(d,rho):
    z=np.log(rho)-np.log1p(-rho)
    local=np.logaddexp(np.log1p(-rho)[None,:],np.log(rho)[None,:]+d.lf.to_numpy()[:,None])
    starts=np.r_[0,np.flatnonzero(np.diff(d.slot.to_numpy()))+1]
    return np.add.reduceat(local,starts,axis=0),starts,z

def score(par,d,n=160):
    rho,lw=quadrature(float(expit(par[0])),float(np.exp(par[1])),n)
    ll,_,_=pair_tables(d,rho)
    return float(logsumexp(ll+lw,axis=1).sum())

def fit(d,n=160):
    r=minimize(lambda p:-score(p,d,n),[1.13017668,.88337189],method='L-BFGS-B',
               bounds=[(-3,3),(-1,8)],options={'maxiter':150,'ftol':1e-10})
    if not r.success: raise RuntimeError(r.message)
    return r.x

def first_k_at_nodes(q,k=5):
    """Columns are quadrature nodes; conditional Bernoulli event process."""
    state=np.zeros((q.shape[1],k));state[:,0]=1
    out=np.zeros_like(q)
    for i,p in enumerate(q):
        out[i]=p*state.sum(axis=1)
        old=state.copy();state*=1-p[:,None];state[:,1:]+=old[:,:-1]*p[:,None]
    return out

def posterior(d,par,n=160):
    rho,lw=quadrature(float(expit(par[0])),float(np.exp(par[1])),n)
    ll,starts,z=pair_tables(d,rho);ends=np.r_[starts[1:],len(d)]
    rows=[];out=d.copy();a=np.zeros((len(d),6))
    for j,(lo,hi) in enumerate(zip(starts,ends)):
        g=d.iloc[lo:hi];postw=np.exp(lw+ll[j]-logsumexp(lw+ll[j]))
        qp=expit(z[None,:]+g.lf.to_numpy()[:,None])
        qa=qp*(-np.expm1(np.minimum((g.l0-g.lf).to_numpy(),0)))[:,None]
        success=g.success.to_numpy() if 'success' in g else np.ones(len(g))
        if not np.isin(success,[0,1]).all(): raise ValueError('success must be a verified 0/1 condition')
        # Integrate AFTER first-K DP, preserving cross-hand dependence via shared rho.
        mp=first_k_at_nodes(qp*success[:,None])@postw
        ma=first_k_at_nodes(qa*success[:,None])@postw
        qpm=qp@postw;qam=qa@postw
        naive=first_k_at_nodes((qam*success)[:,None])[:,0]
        a[lo:hi]=np.c_[qpm,qam,mp,ma,naive,np.repeat(rho@postw,len(g))]
        cdf=np.cumsum(postw)
        rows.append({'slot':int(g.slot.iloc[0]),'rho_mean':float(rho@postw),
                     'rho_q025':float(rho[np.searchsorted(cdf,.025)]),
                     'rho_q975':float(rho[min(np.searchsorted(cdf,.975),len(rho)-1)]),
                     'pair_llr':float(logsumexp(lw+ll[j])),'hands':len(g)})
    out[['q_plant_re','q_act_re','first5_plant_re','first5_act_re','first5_act_naive','rho_pair_mean']]=a
    return out,pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--nodes',type=int,default=160)
    p.add_argument('--cv-seeds',default='260919,11,29,47');p.add_argument('--fit-only',action='store_true')
    args=p.parse_args();d=validate(pd.read_csv(args.input) if args.input.name.endswith(('.csv','.csv.gz')) else pd.read_parquet(args.input))
    args.out.mkdir(parents=True,exist_ok=True);cv=[]
    if not args.fit_only:
        for seed in map(int,args.cv_seeds.split(',')):
            total=0.;folds=[]
            for f,(tr,va) in enumerate(GroupKFold(5,shuffle=True,random_state=seed).split(d,groups=d.pool)):
                train=d.iloc[tr].reset_index(drop=True);valid=d.iloc[va].reset_index(drop=True)
                par=fit(train,args.nodes);v=score(par,valid,args.nodes);total+=v
                folds.append({'fold':f,'llr':v,'mu':float(expit(par[0])),'kappa':float(np.exp(par[1]))})
            cv.append({'seed':seed,'heldout_llr':total,'folds':folds});print(json.dumps(cv[-1]),flush=True)
    par=fit(d,args.nodes);out,pairs=posterior(d,par,args.nodes)
    out.to_csv(args.out/'random_effects_hand_posteriors.csv.gz',index=False)
    pairs.to_csv(args.out/'random_effects_pair_posteriors.csv',index=False)
    fitdata={'mu':float(expit(par[0])),'kappa':float(np.exp(par[1])),'params':par.tolist(),
             'full_llr':score(par,d,args.nodes),'double_nodes_llr':score(par,d,2*args.nodes),
             'cv':cv,'pairs':int(d.slot.nunique()),'hands':len(d),'pools':int(d.pool.nunique()),
             'success_condition_supplied':'success' in d,
             'status':'selected-member, frozen-upstream held-out likelihood only; no evidence MAP or LB gain'}
    (args.out/'random_effects_fit.json').write_text(json.dumps(fitdata,indent=2));print(json.dumps(fitdata),flush=True)
if __name__=='__main__':main()
