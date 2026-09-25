"""Conditional normal-vs-sparse-deviation likelihood features.

Research prototype, not a trained competition model. If the estimated normal
policy is misspecified, these are anomaly FEATURES, not calibrated p/e-values.
Chronology refers only to actual gameplay opportunities, never annotation order.
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit, logsumexp

def log_ratios(y: np.ndarray,p: np.ndarray,delta: float) -> np.ndarray:
    y,p=np.asarray(y,dtype=float),np.asarray(p,dtype=float)
    if y.shape!=p.shape or y.ndim!=1 or not np.isin(y,[0,1]).all(): raise ValueError('Binary vectors required')
    if not np.isfinite(p).all() or np.any((p<=0)|(p>=1)): raise ValueError('Probabilities must lie in (0,1)')
    # Stable Bernoulli log-probability under a fixed log-odds tilt.
    lp=np.log(p)-np.log1p(-p);lq=lp+float(delta)
    return y*(lq-lp)-np.logaddexp(0,lq)+np.logaddexp(0,lp)

def sparse_mixture(y: np.ndarray,p: np.ndarray,deltas=(.7,1.4,2.8),rhos=(.02,.08,.25)) -> dict:
    """Integrate fixed alternatives, not maximize them on evaluated labels.

Each hand can be active with rho. The posterior is over latent behavioral
activation; it is NOT a probability of inclusion in the reference evidence list.
"""
    if not deltas or not rhos or any(not 0<r<1 for r in rhos): raise ValueError('Invalid mixture grid')
    evidences=[];post=[]
    for delta in deltas:
        ell=log_ratios(y,p,delta)
        for rho in rhos:
            terms=np.logaddexp(np.log1p(-rho),np.log(rho)+ell)
            evidences.append(terms.sum())
            post.append(np.exp(np.log(rho)+ell-terms))
    z=np.asarray(evidences);weights=np.exp(z-logsumexp(z))
    return {'log_bf':float(logsumexp(z)-np.log(len(z))),
            'activation':weights@np.asarray(post),'component_log_bf':z,'component_weights':weights}

def episodic_hmm(y: np.ndarray,p: np.ndarray,delta: float=1.4,on_rate: float=.03,off_rate: float=.15) -> dict:
    """Two-state HMM in gameplay-opportunity time; stationary initial state.

A gap of 1 means the next eligible opportunity, not a unit of clock time.
For irregular real-time effects, supply a time-aware transition separately.
"""
    if not 0<on_rate<1 or not 0<off_rate<1:raise ValueError('Transition probabilities outside (0,1)')
    ell=log_ratios(y,p,delta);n=len(ell)
    if n==0:return {'log_bf':0.,'activation':np.zeros(0)}
    transition=np.log([[1-on_rate,on_rate],[off_rate,1-off_rate]])
    initial=np.log([off_rate/(on_rate+off_rate),on_rate/(on_rate+off_rate)])
    emission=np.column_stack([np.zeros(n),ell]);forward=np.empty((n,2));forward[0]=initial+emission[0]
    for t in range(1,n):forward[t]=logsumexp(forward[t-1,:,None]+transition,axis=0)+emission[t]
    total=logsumexp(forward[-1]);back=np.zeros((n,2))
    for t in range(n-2,-1,-1):back[t]=logsumexp(transition+emission[t+1][None,:]+back[t+1][None,:],axis=1)
    return {'log_bf':float(total),'activation':np.exp(forward[:,1]+back[:,1]-total)}
