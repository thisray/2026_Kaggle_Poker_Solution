"""Fixed two-view CatBoost residual probe for GB10.

Outputs only development predictions. This script does NOT make full nested
upstream features: --protocol must explicitly acknowledge frozen-upstream
exploration. Promote only after a separate end-to-end nested confirmation.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression


def run(pack, oof, out, protocol, seed=20260918, threads=8):
    if protocol != 'frozen_upstream_exploration': raise ValueError('Unsupported validation claim')
    root=Path(pack);out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(root/'meta.csv');x=np.load(root/'witness_views.npy',mmap_mode='r')
    if len(d)!=len(x) or x.ndim!=3 or x.shape[1]!=2:raise ValueError('Pack shape mismatch')
    if d.duplicated(['slot','hand_id']).any():raise ValueError('Duplicate candidate key')
    if d.groupby('pool').fold.nunique().max()>1:raise ValueError('Pool crosses folds')
    s=pd.read_csv(oof)
    if s.duplicated(['slot','hand_id']).any() or len(s)!=len(d):
        raise ValueError('Use the full development pack matching this OOF')
    order=pd.MultiIndex.from_frame(d[['slot','hand_id']]).get_indexer(pd.MultiIndex.from_frame(s[['slot','hand_id']]))
    if (order<0).any():raise ValueError('OOF key missing in feature pack')
    # Preserve the original OOF tie order; the pack builder may sort its rows.
    d=d.iloc[order].reset_index(drop=True);x=np.asarray(x[order])
    merged=d[['slot','hand_id']].merge(s,on=['slot','hand_id'],how='left',validate='one_to_one')
    base=merged.rs_blend.to_numpy(float)
    if not np.isfinite(base).all():raise ValueError('OOF alignment failed')
    if not np.array_equal(d.ev.to_numpy(int),merged.ev.to_numpy(int)):raise ValueError('Truth alignment mismatch')
    y=d.ev.to_numpy(int);fold=d.fold.to_numpy();pred=np.zeros(len(d))
    config={'scope':protocol,'seed':seed,'iterations':400,'depth':5,'learning_rate':.03,
            'l2_leaf_reg':30.,'residual_weight':.25,'threads':threads,
            'feature_encoder':'role/street event witnesses, score residual; average two predictions'}
    (out/'preregistered_config.json').write_text(json.dumps(config,indent=2))
    fold_reports=[]
    for f in sorted(np.unique(fold)):
        tr=fold!=f;va=~tr
        xx=np.asarray(x[tr]).reshape(-1,x.shape[-1]);xv=np.asarray(x[va]).reshape(-1,x.shape[-1])
        # Train-only constant screening; no label-based feature selection here.
        keep=np.ptp(xx,axis=0)>0
        xx=xx[:,keep];xv=xv[:,keep]
        cal=LogisticRegression(C=10.,max_iter=1000).fit(base[tr,None],y[tr])
        slope=float(cal.coef_[0,0]);intercept=float(cal.intercept_[0])
        if slope<=1e-8:raise ValueError('Baseline calibration slope is not positive')
        b=slope*base[tr]+intercept
        model=CatBoostClassifier(iterations=400,depth=5,learning_rate=.03,l2_leaf_reg=30,
              random_seed=seed,thread_count=threads,verbose=False,allow_writing_files=False)
        weights=.5/np.minimum(d.m_p.to_numpy(float)[tr],5)
        model.fit(Pool(xx,np.repeat(y[tr],2),baseline=np.repeat(b,2),weight=np.repeat(weights,2)))
        residual=model.predict(xv,prediction_type='RawFormulaVal').reshape(-1,2).mean(1)
        pred[va]=base[va]+.25*residual/slope
        model.save_model(str(out/f'fold_{f}.cbm'));np.save(out/f'fold_{f}_feature_mask.npy',keep)
        fold_reports.append({'fold':int(f),'retained_features':int(keep.sum()),'slope':slope,'intercept':intercept})
    # Independent implementation, fixed original full-truth denominator.
    def scores(values):
        rows=[]
        for slot,ix in d.groupby('slot',sort=False).indices.items():
            yy=y[ix][np.argsort(-values[ix],kind='stable')][:5]
            rows.append({'slot':slot,'pool':d.pool.iloc[ix[0]],'fold':int(fold[ix[0]]),
                'E':float(np.sum(yy*np.cumsum(yy)/np.arange(1,len(yy)+1))/min(int(d.m_p.iloc[ix[0]]),5))})
        return pd.DataFrame(rows)
    bm,pm=scores(base),scores(pred);bm.to_csv(out/'baseline_pairs.csv',index=False);pm.to_csv(out/'candidate_pairs.csv',index=False)
    z=d[['slot','hand_id','pool','fold','ev','m_p']].copy();z['base']=base;z['candidate']=pred
    z.to_csv(out/'oof_predictions.csv.gz',index=False)
    report={'evidence_state':'EXECUTED_FROZEN_UPSTREAM_EXPLORATION','base_E':float(bm.E.mean()),
            'candidate_E':float(pm.E.mean()),'delta_E':float(pm.E.mean()-bm.E.mean()),'folds':fold_reports,
            'pack_receipt':json.loads((root/'pack.json').read_text()),
            'oof_sha256':hashlib.sha256(Path(oof).read_bytes()).hexdigest(),
            'promotion':'No automatic READY, no Kaggle submission. Requires separate nested confirmation.'}
    (out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['pack','oof','out','protocol']:p.add_argument('--'+n,required=True)
    p.add_argument('--seed',type=int,default=20260918);p.add_argument('--threads',type=int,default=8)
    run(**vars(p.parse_args()))
