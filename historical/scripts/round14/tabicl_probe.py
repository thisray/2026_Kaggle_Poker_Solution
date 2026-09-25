"""Optional GB10 pre-trained tabular model probe; not run in review environment.
Official API checked 2026-09-18. Supply a locally verified checkpoint explicitly.
No network download, no validation-label early stopping, no Kaggle submission.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from pair_comparator_probe import SCORES,ENTRY,pair_metrics,bootstrap

def run(input,checkpoint,out,include_entry=False,device='cuda'):
    from tabicl import TabICLClassifier
    if not Path(checkpoint).is_file():raise FileNotFoundError('Verified local checkpoint required')
    d=pd.read_csv(input);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cols=SCORES+(ENTRY if include_entry else []);x=d[cols].to_numpy(float);r=np.empty_like(x)
    assert d.groupby('pool').fold.nunique().max()==1
    for ix in d.groupby('slot').indices.values():
        r[ix]=np.column_stack([rankdata(x[ix,j])/len(ix) for j in range(x.shape[1])])
    x=np.nan_to_num(np.c_[x,r],nan=0.,posinf=30,neginf=-30).clip(-30,30);pred=np.zeros(len(d))
    for f in sorted(d.fold.unique()):
        tr=d.fold.to_numpy()!=f;va=~tr
        model=TabICLClassifier(n_estimators=4,model_path=checkpoint,allow_auto_download=False,device=device,random_state=20260918)
        model.fit(x[tr],d.ev.to_numpy()[tr]);pred[va]=model.predict_proba(x[va])[:,1]
        print('completed fold',f,flush=True)
    bm=pair_metrics(d,d.rs_blend.to_numpy());pm=pair_metrics(d,pred)
    result={'protocol':'frozen_upstream_exploration','checkpoint':str(checkpoint),'columns':cols,'E':float(pm.E.mean()),**bootstrap(bm,pm)}
    (out/'results.json').write_text(json.dumps(result,indent=2));d[['slot','hand_id','pool','fold','ev','m_p']].assign(candidate=pred).to_csv(out/'oof.csv.gz',index=False)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for x in ['input','checkpoint','out']:p.add_argument('--'+x,required=True)
    p.add_argument('--include-entry',action='store_true');p.add_argument('--device',default='cuda');run(**vars(p.parse_args()))
