#!/usr/bin/env python3
"""Fit R3-style hierarchy from cached decision likelihood ratios, no policy calls.
Use the original member flag in t17_bf_sub_eval as calibration to reproduce R3.
This adapter is source-reviewed; real GB10 cached rows were unavailable here.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,log_expit
p=argparse.ArgumentParser();p.add_argument('--rows',required=True);p.add_argument('--members',required=True);p.add_argument('--out',required=True);a=p.parse_args()
m=pd.read_parquet(a.members)
if not {'slot','member'}<=set(m):raise ValueError('Need R3 slot/member calibration table')
slots=set(m.loc[m.member.fillna(False).astype(bool),'slot'])
r=pd.read_parquet(a.rows,columns=['slot','h','st','r']);r=r[r.slot.isin(slots)].copy()
if r.empty or not np.isfinite(r.r).all() or (r.r<0).any():raise ValueError('Invalid cached calibration rows')
inv=pd.MultiIndex.from_frame(r[['slot','h']]).factorize()[0];pre=r.st.to_numpy()==0;rr=r.r.to_numpy(float)
def nll(th):
    s=np.where(pre,expit(th[1]),expit(th[2]));s=np.clip(s,1e-12,1-1e-12)
    lg=np.bincount(inv,weights=np.log1p(s*(rr-1)))
    return -np.logaddexp(log_expit(-th[0]),log_expit(th[0])+lg).sum()
fits=[minimize(nll,np.array(t,float),method='Nelder-Mead',options={'maxiter':4000}) for t in [[1,0,-.3],[0,.5,0],[2,-.5,-.5]]]
f=min(fits,key=lambda z:z.fun);rho,sp,sq=expit(f.x)
z={'rho':float(rho),'s_pre':float(sp),'s_post':float(sq),'log_likelihood':float(-f.fun),'success':bool(f.success),'n_calibration_members':len(slots),'n_rows':len(r),'status':'Refitted from cached ratios; compare q_act/q_plant against R3 per-hand table before use.'}
Path(a.out).write_text(json.dumps(z,indent=2));print(json.dumps(z,indent=2))
