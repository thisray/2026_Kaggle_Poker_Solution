"""R5: apply an evidence patch to a base candidate at an arbitrary path.
Same contract as R4's y5_build_candidates.py (risk and predicted_behavior are never
touched; a patch row is applied only if the base routes that pair to the patch's
family) but the base may live outside r2_candidates.
usage: python y5r5_build.py <base.csv full path> <out.csv full path> <family> <patch.csv> [...]"""
import json,hashlib,sys,os
import pandas as pd
EVC=[f'evidence_hand_{i}' for i in range(1,6)]
base_path,out_path,fam=sys.argv[1],sys.argv[2],sys.argv[3]; patches=sys.argv[4:]
base=pd.read_csv(base_path,dtype=str,keep_default_na=False); out=base.set_index('pair_id').copy()
rec=dict(file=os.path.basename(out_path),base=os.path.basename(base_path),base_sha=hashlib.sha256(open(base_path,'rb').read()).hexdigest(),patches=[])
for p in patches:
    pt=pd.read_csv(p,dtype=str,keep_default_na=False)
    ok=pt.pair_id.isin(out.index[out.predicted_behavior==fam]); keep=pt[ok]
    before=out.loc[keep.pair_id,EVC].values.copy(); out.loc[keep.pair_id,EVC]=keep.set_index('pair_id')[EVC].values
    rec['patches'].append(dict(patch=os.path.basename(p),family=fam,rows=len(pt),applied=int(ok.sum()),skipped_not_routed=int((~ok).sum()),
        pairs_with_changed_set=int(sum(set(a)!=set(b) for a,b in zip(before,out.loc[keep.pair_id,EVC].values)))))
o=out.reset_index()[base.columns]; o.to_csv(out_path,index=False)
assert (o.risk_score.values==base.risk_score.values).all() and (o.predicted_behavior.values==base.predicted_behavior.values).all() and (o.pair_id.values==base.pair_id.values).all()
rec.update(sha256=hashlib.sha256(open(out_path,'rb').read()).hexdigest(),evidence_rows_changed=int((o[EVC].values!=base[EVC].values).any(1).sum()),risk_identical=True,behavior_identical=True)
json.dump(rec,open(out_path.replace('.csv','.receipt.json'),'w'),indent=1); print(json.dumps(rec,indent=1))
