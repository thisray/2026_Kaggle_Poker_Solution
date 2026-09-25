"""Context-gated hierarchical partner-card mixture over cached likelihood ratios.

Run: python f4_decision_gate.py --data cache.npz --out fitted.npz
NPZ requires Xh (H x P), Xd (D x Q), hand (D contiguous indices in [0,H)),
log_ratio (D): log policy_partner(observed_action)/policy_own(observed_action).
First column of both feature matrices must be intercept 1. All covariates
must be available BEFORE the decision; exclude action taken / future outcomes.
Use preflop indicator and postflop indicator with care to avoid collinearity.
Fitting all rows does not constitute held-out validation. Worker must fit train
pools and score other pools; normal-policy fitting and member selection are
upstream and require separate controls. This script never submits anything.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize


def components(theta,Xh,Xd,hand,log_ratio):
    ph=Xh.shape[1];zh=Xh@theta[:ph];zd=Xd@theta[ph:]
    # stable log[1-s+s*r] and posterior P(active|plant,observed decision)
    ld=np.logaddexp(0,zd+log_ratio)-np.logaddexp(0,zd)
    lf=np.bincount(hand,weights=ld,minlength=len(Xh))
    l0=np.bincount(hand,weights=-np.logaddexp(0,zd),minlength=len(Xh))
    rho=expit(zh);qplant=expit(zh+lf);s=expit(zd)
    active_given_plant=expit(zd+log_ratio)
    hand_llr=np.logaddexp(0,zh+lf)-np.logaddexp(0,zh)
    qact=qplant*(-np.expm1(np.minimum(l0-lf,0)))
    return hand_llr,qplant,qact,rho,s,active_given_plant


def objective(theta,Xh,Xd,hand,log_ratio,l2=10.):
    ll,qp,qa,rho,s,active=components(theta,Xh,Xd,hand,log_ratio)
    ph=Xh.shape[1];penalty=np.ones(len(theta));penalty[[0,ph]]=0
    f=-ll.sum()+.5*l2*np.dot(theta*penalty,theta)
    gh=-Xh.T@(qp-rho)
    gd=-Xd.T@(qp[hand]*(active-s))
    g=np.r_[gh,gd]+l2*penalty*theta
    return float(f),g


def fit(Xh,Xd,hand,log_ratio,l2=10.,initial=None):
    Xh=np.asarray(Xh,float);Xd=np.asarray(Xd,float);hand=np.asarray(hand,int)
    lr=np.asarray(log_ratio,float)
    if Xh.ndim!=2 or Xd.ndim!=2 or len(hand)!=len(Xd) or len(lr)!=len(Xd):
        raise ValueError('Incompatible matrix dimensions')
    if hand.min()<0 or hand.max()>=len(Xh): raise ValueError('Invalid hand indices')
    if not all(np.isfinite(x).all() for x in [Xh,Xd,lr]):raise ValueError('Nonfinite input')
    if not np.allclose(Xh[:,0],1) or not np.allclose(Xd[:,0],1):
        raise ValueError('Column zero must be an intercept')
    init=np.zeros(Xh.shape[1]+Xd.shape[1]) if initial is None else np.asarray(initial,float)
    if initial is None:init[0]=.89 # roughly old rho, s starts .5
    opt=minimize(objective,init,args=(Xh,Xd,hand,lr,l2),jac=True,method='L-BFGS-B',
                 bounds=[(-7,7)]*len(init),options={'maxiter':500,'ftol':1e-10})
    if not opt.success:raise RuntimeError(str(opt.message))
    return opt.x


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--l2',type=float,default=10.);a=p.parse_args()
    d=np.load(a.data,allow_pickle=False);names=['Xh','Xd','hand','log_ratio']
    arrays=[d[k] for k in names];b=fit(*arrays,l2=a.l2)
    ll,qp,qa,rho,s,_=components(b,*arrays)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,coef=b,q_plant=qp,q_act=qa,rho=rho,s=s,hand_llr=ll)
    print(json.dumps({'hands':len(qp),'decisions':len(s),'fit_llr':float(ll.sum()),
                     'status':'fitted only; perform held-out/negative-control evaluation separately'}))
if __name__=='__main__':main()
