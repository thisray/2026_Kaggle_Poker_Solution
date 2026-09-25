"""Logistic ranking lambdas weighted by exact competition AP@5 swap deltas.
Use full planted truth count in the denominator, never candidate positives.
Exact metric weights still define a surrogate, not a differentiable exact AP loss.
"""
import numpy as np
from numba import njit
@njit(cache=True)
def ap_at_5(y,denom):
    hit=0.;res=0.
    for i in range(min(5,len(y))):
        if y[i]>0:hit+=1.;res+=hit/(i+1)
    return res/denom
@njit(cache=True)
def swap_delta(y,a,b,denom):
    base=ap_at_5(y,denom);temp=y.copy();temp[a],temp[b]=temp[b],temp[a]
    return abs(ap_at_5(temp,denom)-base)
@njit(cache=True)
def lambdas(y,pred,starts,denoms):
    grad=np.zeros(len(y));hess=np.zeros(len(y))+1e-8
    for q in range(len(denoms)):
        s,e=starts[q],starts[q+1];ix=np.argsort(-pred[s:e],kind='mergesort');yy=y[s:e][ix]
        for a in range(e-s):
            if yy[a]!=1:continue
            for b in range(e-s):
                if yy[b]!=0:continue
                w=swap_delta(yy,a,b,denoms[q])
                if w<=0:continue
                ip=s+ix[a];ineg=s+ix[b];v=min(40.,max(-40.,pred[ip]-pred[ineg]));p=1./(1.+np.exp(v))
                g=w*p;h=w*p*(1-p);grad[ip]-=g;grad[ineg]+=g;hess[ip]+=h;hess[ineg]+=h
    return grad,hess

def objective_for(groups,full_truth_count):
    groups=np.asarray(groups,dtype=np.int64);starts=np.r_[0,np.cumsum(groups)].astype(np.int64)
    denoms=np.minimum(np.asarray(full_truth_count,float),5.)
    if len(groups)!=len(denoms) or np.any(denoms<=0):raise ValueError('Invalid query truth')
    def objective(pred,data):return lambdas(data.get_label().astype(np.float64),np.asarray(pred,float),starts,denoms)
    return objective
