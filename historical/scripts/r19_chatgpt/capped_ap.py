"""Exact expected AP@K moments for the first K independent Bernoulli events.
No hidden labels, network calls, or uploads. Candidate indexes refer to the supplied
chronological event-probability vector; they are never predictive ID features.
"""
from __future__ import annotations
import numpy as np
from numba import njit

@njit(cache=True)
def _advance(a,p):
    b=a*(1-p)
    for k in range(1,len(a)): b[k]+=a[k-1]*p
    b[-1]+=a[-1]*p
    return b

@njit(cache=True)
def _moments(p,candidates,k,constant_denominator):
    n=len(p);m=len(candidates)
    pre=np.zeros((n+1,k+1));suf=np.zeros((n+1,k+1))
    pre[0,0]=1.;suf[n,0]=1.
    for t in range(n):pre[t+1]=_advance(pre[t],p[t])
    for t in range(n-1,-1,-1):suf[t]=_advance(suf[t+1],p[t])
    a=np.zeros(m);b=np.zeros((m,m))
    for u in range(m):
        i=candidates[u]
        for c in range(k):
            for d in range(k+1):
                den=k if constant_denominator else min(k,c+d+1)
                a[u]+=p[i]*pre[i,c]*suf[i+1,d]/den
        b[u,u]=a[u]
        excl=pre[i].copy();last=i+1
        for v in range(u+1,m):
            j=candidates[v]
            while last<j:
                excl=_advance(excl,p[last]);last+=1
            value=0.
            for c in range(k-1):
                for d in range(k+1):
                    den=k if constant_denominator else min(k,c+d+2)
                    value+=p[i]*p[j]*excl[c]*suf[j+1,d]/den
            b[u,v]=value;b[v,u]=value
    return a,b

def moments(p,candidates,k=5,constant_denominator=False):
    p=np.asarray(p,dtype=float);c=np.asarray(candidates,dtype=np.int64)
    if p.ndim!=1 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('p must be a finite 1D probability array')
    if k<1 or np.any(c<0) or np.any(c>=len(p)) or np.any(np.diff(c)<=0):
        raise ValueError('Candidate positions must be unique and strictly increasing')
    return _moments(p,c,k,constant_denominator)

@njit(cache=True)
def _optimize(a,b,k):
    m=len(a);lim=1<<m
    dp=np.full(lim,-1e100);dp[0]=0.;count=np.zeros(lim,np.int8);last=np.full(lim,-1,np.int16)
    best=-1e100;bestmask=0
    for mask in range(1,lim):
        count[mask]=count[mask>>1]+(mask&1)
        r=count[mask]
        if r>k:continue
        for j in range(m):
            if mask&(1<<j):
                prev=mask^(1<<j);value=a[j]
                for i in range(m):
                    if prev&(1<<i):value+=b[i,j]
                value=dp[prev]+value/r
                if value>dp[mask]:dp[mask]=value;last[mask]=j
        if r==k and dp[mask]>best:best=dp[mask];bestmask=mask
    out=np.empty(k,np.int64);mask=bestmask
    for r in range(k-1,-1,-1):out[r]=last[mask];mask^=1<<out[r]
    return out,best

def optimize(a,b,k=5):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if len(a)>18:raise ValueError('Use <=18 candidates to bound exact subset DP')
    if len(a)<k or b.shape!=(len(a),len(a)) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid moments or too few candidates')
    return _optimize(a,b,k)

def expected_ap(order,a,b):
    return float(sum((a[j]+sum(b[i,j] for i in order[:r]))/(r+1) for r,j in enumerate(order)))
