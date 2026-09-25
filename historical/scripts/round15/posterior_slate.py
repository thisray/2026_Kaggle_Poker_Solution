"""Optimize expected AP@K under a gameplay-only posterior over relevance sets.
An exact expected utility identity, NOT a posterior estimator or a score claim.
K<=5 and N<=20 recommended. Never feed evaluation truth / planted-label order.
"""
from __future__ import annotations
import argparse, json
from itertools import combinations
from pathlib import Path
import numpy as np


def coefficients(relevance, weights=None, k=5, full_counts=None):
    y=np.asarray(relevance,float)
    if y.ndim!=2 or not np.isin(y,[0,1]).all():raise ValueError('Binary posterior scenarios needed')
    w=np.ones(len(y))/len(y) if weights is None else np.asarray(weights,float)
    if len(w)!=len(y) or np.any(w<0) or not np.isfinite(w).all() or w.sum()<=0:raise ValueError('Weights')
    w=w/w.sum();m=y.sum(1) if full_counts is None else np.asarray(full_counts,float)
    if np.any(m<y.sum(1)) or np.any(m<0):raise ValueError('Full relevant count must include candidates')
    inv=np.divide(w,np.minimum(m,k),out=np.zeros_like(w),where=m>0)
    single=(inv[:,None]*y).sum(0)
    joint=(y.T*inv)@y;np.fill_diagonal(joint,0.)
    return single,joint


def expected_score(order,single,joint):
    return float(sum((single[i]+sum(joint[i,j] for j in order[:r]))/(r+1) for r,i in enumerate(order)))


def optimize(single,joint,k=5):
    """Exact subset DP; cost depends on N choose K. No hand IDs are used."""
    a=np.asarray(single);b=np.asarray(joint);n=len(a);k=min(k,n)
    if n>24:raise ValueError('Use a fixed gameplay-based candidate shortlist <=24')
    dp={0:(0.,())}
    for size in range(1,k+1):
        level={}
        for subset in combinations(range(n),size):
            bits=sum(1<<i for i in subset);best=(-np.inf,())
            for i in subset:
                old,order=dp[bits^(1<<i)]
                val=old+(a[i]+sum(b[i,j] for j in subset if j!=i))/size
                if val>best[0]+1e-15:best=(float(val),order+(i,))
            level[bits]=best
        dp=level
    value,order=max(dp.values(),key=lambda z:z[0]);return list(order),value


def scenario_ap(order,y,counts=None,k=5):
    y=np.asarray(y);m=y.sum(1) if counts is None else np.asarray(counts)
    top=y[:,order[:k]];num=(top*np.cumsum(top,1)/np.arange(1,top.shape[1]+1)).sum(1)
    return np.divide(num,np.minimum(m,k),out=np.zeros_like(num,dtype=float),where=m>0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--posterior',required=True);p.add_argument('--out',required=True)
    p.add_argument('--k',type=int,default=5);a=p.parse_args()
    z=np.load(a.posterior,allow_pickle=False)
    s,j=coefficients(z['relevance'],z.get('weights'),a.k,z.get('full_counts'))
    order,value=optimize(s,j,a.k)
    Path(a.out).write_text(json.dumps({'order':order,'posterior_expected_AP':value,
            'state':'posterior_model_output_not_empirical_validation'},indent=2))
