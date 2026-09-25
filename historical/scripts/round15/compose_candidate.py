"""Compose authorized local prediction artifacts; never submit to Kaggle.
Risk/B/E sources can differ. This only checks schema; reuse existing membership
validator on GB10 before marking READY. It does NOT infer any labels.
"""
import argparse,json,hashlib
from pathlib import Path
import pandas as pd
import numpy as np

def run(template,risk,behavior,evidence,out):
    t=pd.read_csv(template);keys=t.pair_id
    if keys.duplicated().any():raise ValueError('template keys')
    ans=t[['pair_id']].copy()
    for path,cols in [(risk,['risk_score']),(behavior,['predicted_behavior']),
                      (evidence,[f'evidence_hand_{i}' for i in range(1,6)])]:
        d=pd.read_csv(path)
        if d.pair_id.duplicated().any() or set(d.pair_id)!=set(keys):raise ValueError('Coverage')
        ans=ans.merge(d[['pair_id']+cols],on='pair_id',validate='one_to_one',sort=False)
    r=ans.risk_score.to_numpy()
    if not np.isfinite(r).all() or not ((r>=0)&(r<=1)).all():raise ValueError('risk range')
    allowed={'none','directed_transfer','soft_play','coordinated_isolation','other_coordination'}
    if not ans.predicted_behavior.isin(allowed).all() or ans.isna().any().any():raise ValueError('labels/empty')
    for row in ans[[f'evidence_hand_{i}' for i in range(1,6)]].itertuples(index=False,name=None):
        h=[v for v in row if v!='NO_EVIDENCE']
        if len(h)!=len(set(h)):raise ValueError('Duplicate evidence')
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    ans[t.columns].to_csv(out,index=False)
    receipt={'file':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
        'state':'LOCAL_COMPOSED_REQUIRES_EXISTING_MEMBERSHIP_VALIDATOR','submitted':False,
        'risk_source':risk,'behavior_source':behavior,'evidence_source':evidence}
    out.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['template','risk','behavior','evidence','out']:p.add_argument('--'+n,required=True)
    run(**vars(p.parse_args()))
