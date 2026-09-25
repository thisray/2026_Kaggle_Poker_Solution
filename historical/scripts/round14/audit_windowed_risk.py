"""Audit original P OOF by window; never deduplicate different windows together."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def ap_prefix(y,score,key):
    order=np.lexsort((np.asarray(key).astype(str),-np.asarray(score,float)))
    t=np.asarray(y,int)[order];p=t*np.cumsum(t)/np.arange(1,len(t)+1)
    return (float(p.sum()/t.sum()) if t.sum() else None),order,p

def run(rowwise,out):
    d=pd.read_parquet(rowwise) if rowwise.endswith('.parquet') else pd.read_csv(rowwise)
    required={'src','key','pool','fold','y','label','base_score','new_score'}
    if required-set(d):raise ValueError(f'Missing: {required-set(d)}')
    if d.duplicated(['src','key']).any():raise ValueError('Duplicate source/key; do not silently keep first')
    if d.groupby('pool').fold.nunique().max()!=1:raise ValueError('Pool spans folds')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);result={}
    for src,b in d.groupby('src',sort=False):
        b=b.sort_values('key',kind='stable').reset_index(drop=True);b.to_csv(out/f'{src}_rowwise.csv.gz',index=False)
        info={'rows':len(b),'pools':int(b.pool.nunique()),'positive_labels':int((b.label==1).sum()),'unknown_labels':int((b.label==-1).sum())}
        if not str(src).startswith('dev'):
            info['state']='COVERAGE_ONLY: evaluation labels unavailable; y=0 is a training assumption'
            result[str(src)]=info;continue
        y=(b.label.to_numpy()==1).astype(int)
        if not np.array_equal(y,b.y.to_numpy()):raise ValueError('Development y disagrees with provided labels')
        for col in ('base_score','new_score'):
            if not np.isfinite(b[col]).all():raise ValueError('Nonfinite risk')
            ap,order,contrib=ap_prefix(y,b[col],b.key)
            info[col]={'AP_U_as_negative_proxy':ap,'known_P_topK':{str(k):int(y[order[:k]].sum()) for k in [100,300,450,1000,2000]}}
        info['delta']=info['new_score']['AP_U_as_negative_proxy']-info['base_score']['AP_U_as_negative_proxy']
        result[str(src)]=info
    (out/'summary.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rowwise',required=True);p.add_argument('--out',required=True);print(json.dumps(run(**vars(p.parse_args())),indent=2))
