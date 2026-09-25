"""First-K event marginals conditional on >=K total events.
The assumption is appropriate only for routes with independently supported count
prior. It does not assume there are no events after the Kth observed one.
"""
import numpy as np
from numba import njit
@njit(cache=True)
def _advance(a,p):
    b=a*(1-p)
    for k in range(1,len(a)):b[k]+=a[k-1]*p
    b[-1]+=a[-1]*p
    return b
@njit(cache=True)
def conditional_first_k(p,k=5):
    n=len(p);pre=np.zeros((n+1,k+1));suf=np.zeros((n+1,k+1))
    pre[0,0]=1.;suf[n,0]=1.
    for t in range(n):pre[t+1]=_advance(pre[t],p[t])
    for t in range(n-1,-1,-1):suf[t]=_advance(suf[t+1],p[t])
    z=pre[n,k];q=np.zeros(n)
    if z<1e-15:return q,z
    for i in range(n):
        for c in range(k):
            need=k-1-c
            q[i]+=p[i]*pre[i,c]*suf[i+1,need:].sum()/z
    return q,z
