"""R19 fixed-emission, context-dependent hand activation prior.

Input CSV/Parquet columns: slot,h,pool,eA,eB,lf,l0; optional ts.
'eA/eB' are PRE-FLOP equities of the first acting member and their partner
(c14 definitions), NOT final-board equities or action outcomes.
lf = sum log(1-s+s*r); l0 = sum log(1-s), using R3 frozen emissions.
Pool-held-out likelihood is exploratory: upstream policy/member selection is frozen.
Produces posterior features, NOT a complete Kaggle submission.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import GroupKFold

MODELS=('constant','gap','own_partner','symmetric_gap','quadratic')

def matrix(d: pd.DataFrame, model: str) -> np.ndarray:
    a=(d.eA.to_numpy(dtype=float)-.5)*10
    b=(d.eB.to_numpy(dtype=float)-.5)*10
    xs={'constant':[np.ones(len(d))], 'gap':[np.ones(len(d)),b-a],
        'own_partner':[np.ones(len(d)),a,b],
        'symmetric_gap':[np.ones(len(d)),a+b,np.abs(b-a)],
        'quadratic':[np.ones(len(d)),a,b,(a-b)**2,a*b]}
    if model not in xs: raise ValueError(f'Unknown model {model}')
    x=np.column_stack(xs[model])
    if not np.isfinite(x).all(): raise ValueError('Missing/nonfinite pre-action covariates')
    return x

def llr(beta: np.ndarray,x: np.ndarray,lf: np.ndarray) -> np.ndarray:
    z=x@beta
    return np.logaddexp(0,z+lf)-np.logaddexp(0,z)

def objective(beta: np.ndarray,x: np.ndarray,lf: np.ndarray,l2: float):
    z=x@beta
    f=-llr(beta,x,lf).sum()+.5*l2*np.dot(beta[1:],beta[1:])
    g=-x.T@(expit(z+lf)-expit(z));g[1:]+=l2*beta[1:]
    return float(f),g

def fit(x: np.ndarray,lf: np.ndarray,l2: float=10.,initial_rho: float=.7091877019):
    init=np.zeros(x.shape[1]);init[0]=np.log(initial_rho/(1-initial_rho))
    r=minimize(objective,init,args=(x,lf,l2),jac=True,method='L-BFGS-B',
               bounds=[(-7,7)]*len(init),options={'maxiter':300})
    if not r.success: raise RuntimeError(f'Gate fit failed: {r.message}')
    return r.x

def predict(beta,x,lf,l0):
    rho=expit(x@beta);qp=expit(x@beta+lf)
    # P(no active decision|plant,data)=exp(l0-lf) under independent decisions.
    qa=qp*(-np.expm1(np.minimum(l0-lf,0)))
    return rho,qp,np.clip(qa,0,1)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',choices=MODELS,default='gap')
    p.add_argument('--l2',type=float,default=10.)
    p.add_argument('--seed',type=int,default=260919)
    a=p.parse_args();d=pd.read_csv(a.input) if a.input.name.endswith(('.csv','.csv.gz')) else pd.read_parquet(a.input)
    required={'slot','h','pool','eA','eB','lf','l0'}
    if not required.issubset(d): raise ValueError(f'Missing {required-set(d.columns)}')
    if d.duplicated(['slot','h']).any(): raise ValueError('Duplicate pair-hands')
    if d.pool.nunique()<5: raise ValueError('Need at least five pools')
    if not np.isfinite(d[['lf','l0']].to_numpy()).all():raise ValueError('Nonfinite emissions')
    a.out.mkdir(parents=True,exist_ok=True)
    x=matrix(d,a.model);xc=matrix(d,'constant');lf=d.lf.to_numpy();l0=d.l0.to_numpy()
    oof=np.zeros((len(d),3));rows=[]
    for f,(tr,va) in enumerate(GroupKFold(5,shuffle=True,random_state=a.seed).split(d,groups=d.pool)):
        b=fit(x[tr],lf[tr],a.l2);bc=fit(xc[tr],lf[tr],a.l2)
        oof[va]=np.column_stack(predict(b,x[va],lf[va],l0[va]))
        rows.append({'fold':f,'gate_llr':float(llr(b,x[va],lf[va]).sum()),
                     'constant_llr':float(llr(bc,xc[va],lf[va]).sum())})
    beta=fit(x,lf,a.l2);rho,qp,qa=predict(beta,x,lf,l0)
    keep=[c for c in ['slot','h','ts','pool'] if c in d]
    out=d[keep].copy();out[['rho','q_plant','q_act']]=np.c_[rho,qp,qa]
    out[['rho_oof','q_plant_oof','q_act_oof']]=oof
    out.to_csv(a.out/'hand_posteriors.csv',index=False)
    meta={'model':a.model,'l2':a.l2,'seed':a.seed,'coef':beta.tolist(),'folds':rows,
          'heldout_delta':sum(r['gate_llr']-r['constant_llr'] for r in rows),
          'status':'exploratory held-out likelihood only; no true F4 evidence validation or LB gain',
          'input_rows':len(d),'pairs':int(d.slot.nunique())}
    (a.out/'fit.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
