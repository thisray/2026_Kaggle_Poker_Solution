#!/usr/bin/env python3
"""New F4 hypothesis: observable action change under maximal coupling.
Consumes cached R3 decision likelihood ratios; performs no policy prediction,
network access, or Kaggle submission. This is NOT a verified annotation rule.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from scipy.special import expit

def decode(rows: pd.DataFrame, rho: float, s_pre: float, s_post: float) -> pd.DataFrame:
    required={'slot','h','st','r'}
    if not required <= set(rows): raise ValueError(f'Missing {required-set(rows)}')
    if not (0<rho<1 and 0<s_pre<1 and 0<s_post<1): raise ValueError('Use fitted interior probabilities.')
    if rows.r.isna().any() or (~np.isfinite(rows.r)).any() or (rows.r<0).any(): raise ValueError('Invalid likelihood ratio')
    if 'k' in rows and rows.duplicated(['slot','h','k']).any(): raise ValueError('Duplicate decision key')
    s=np.where(rows.st.to_numpy()==0,s_pre,s_post)
    r=rows.r.to_numpy(dtype=float)
    logg=np.log1p(s*(r-1))
    # Conditional on a planted hand and the observed action:
    # P(Z=1|a)=s*r/g; P(action counterfactually changed|a)=s*(r-1)_+/g.
    log_no_active=np.log1p(-s)-logg
    u=s*np.maximum(r-1,0)/np.exp(logg)
    log_no_witness=np.log1p(-np.minimum(u,1-1e-15))
    z=rows[['slot','h']].copy()
    z['log_g']=logg; z['log_no_active']=log_no_active; z['log_no_witness']=log_no_witness
    g=z.groupby(['slot','h'],sort=False).agg(log_bf=('log_g','sum'),log_no_active=('log_no_active','sum'),log_no_witness=('log_no_witness','sum'),n_decisions=('log_g','size'))
    plant=expit(np.log(rho/(1-rho))+g.log_bf.to_numpy())
    g['q_plant']=plant
    g['q_act']=plant*(-np.expm1(g.log_no_active.to_numpy()))
    g['q_witness']=plant*(-np.expm1(g.log_no_witness.to_numpy()))
    if not ((g.q_witness>=-1e-12)&(g.q_witness<=g.q_act+1e-12)&(g.q_act<=g.q_plant+1e-12)).all():
        raise AssertionError('Posterior ordering failed')
    return g.reset_index()

def self_test() -> dict:
    import itertools
    q0=np.array([[.8,.2],[.6,.4],[.3,.7]])
    qs=np.array([[.2,.8],[.5,.5],[.8,.2]])
    s=np.array([.5,.4,.4]);rho=.7;y=np.array([1,0,1]);r=qs[np.arange(3),y]/q0[np.arange(3),y]
    rows=pd.DataFrame({'slot':[1]*3,'h':[10]*3,'k':range(3),'st':[0,1,1],'r':r})
    ans=decode(rows,rho,s[0],s[1]).iloc[0]
    denom=(1-rho)*np.prod(q0[np.arange(3),y]); active=0.;plant=0.;no_wit=0.
    for z in itertools.product([0,1],repeat=3):
        z=np.array(z);weight=rho*np.prod(np.where(z,s,1-s))*np.prod(np.where(z,qs[np.arange(3),y],q0[np.arange(3),y]))
        denom+=weight;plant+=weight
        if z.any(): active+=weight
        # Conditional no-change mass given substitution action a under maximal coupling.
        factors=np.where(z,np.minimum(q0[np.arange(3),y],qs[np.arange(3),y])/qs[np.arange(3),y],1.)
        no_wit+=weight*np.prod(factors)
    expected=np.array([plant,active,plant-no_wit])/denom
    actual=ans[['q_plant','q_act','q_witness']].to_numpy(dtype=float)
    assert np.allclose(actual,expected,rtol=1e-12)
    null=decode(rows.assign(r=1.),rho,s[0],s[1]).iloc[0]
    assert null.q_witness==0 and null.q_act>0
    return {'enumeration_pass':True,'q_plant':float(actual[0]),'q_act':float(actual[1]),'q_witness':float(actual[2]),'null_witness':float(null.q_witness),'status':'Math-only test. No real evidence or leaderboard gain measured.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--rows');p.add_argument('--out');p.add_argument('--rho',type=float);p.add_argument('--s-pre',type=float);p.add_argument('--s-post',type=float);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test: print(json.dumps(self_test(),indent=2));return
    if not all(x is not None for x in [a.rows,a.out,a.rho,a.s_pre,a.s_post]):p.error('Provide --rows --out and fitted --rho --s-pre --s-post')
    rows=pd.read_csv(a.rows) if a.rows.endswith('.csv') else pd.read_parquet(a.rows)
    out=decode(rows,a.rho,a.s_pre,a.s_post)
    if a.out.endswith('.csv'):out.to_csv(a.out,index=False)
    else:out.to_parquet(a.out,index=False)
if __name__=='__main__':main()
