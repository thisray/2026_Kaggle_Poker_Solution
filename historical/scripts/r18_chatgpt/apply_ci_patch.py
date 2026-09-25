#!/usr/bin/env python3
"""Apply a CI-only evidence patch while preserving all P/B strings exactly.
Does not upload. Pair-hand legality is checked by the raw extraction adapter.
"""
import argparse,json,hashlib
from pathlib import Path
import pandas as pd

def apply(base: pd.DataFrame, patch: pd.DataFrame):
    cols=[f'evidence_hand_{i}' for i in range(1,6)]
    if not {'pair_id',*cols}<=set(patch):raise ValueError('Incomplete patch')
    if not base.pair_id.is_unique or not patch.pair_id.is_unique:raise ValueError('Duplicate pair')
    if not set(patch.pair_id)<=set(base.pair_id):raise ValueError('Unknown pair')
    if patch[cols].isna().any().any():raise ValueError('Empty evidence')
    for v in patch[cols].itertuples(index=False,name=None):
        if 'NO_EVIDENCE' in v or len(set(v))!=5:raise ValueError('Patch must contain five distinct real hands')
    old=base.copy().set_index('pair_id');new=old.copy();ids=patch.pair_id
    if not old.loc[ids,'predicted_behavior'].eq('coordinated_isolation').all():raise ValueError('Patch would touch non-CI pair')
    new.loc[ids,cols]=patch.set_index('pair_id').loc[ids,cols]
    assert new[['risk_score','predicted_behavior']].equals(old[['risk_score','predicted_behavior']])
    assert new.loc[~new.index.isin(ids)].equals(old.loc[~old.index.isin(ids)])
    changed=int(new[cols].ne(old[cols]).any(axis=1).sum())
    return new.reset_index()[base.columns],{'changed_evidence_rows':changed,'changed_P_rows':0,'changed_B_rows':0,'not_submitted':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--patch',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    b=pd.read_csv(a.base,dtype=str,keep_default_na=False);q=pd.read_csv(a.patch,dtype=str,keep_default_na=False)
    out,receipt=apply(b,q);out.to_csv(a.out,index=False)
    receipt['sha256']=hashlib.sha256(Path(a.out).read_bytes()).hexdigest()
    Path(a.out+'.receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
