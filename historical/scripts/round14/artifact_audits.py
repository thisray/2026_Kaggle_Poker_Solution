"""Reproduce bounded artifact diagnostics, without retraining any upstream model."""
from __future__ import annotations
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
import pandas as pd
from pair_comparator_probe import pair_metrics,bootstrap

def geometry(d,col):
    r=[]
    for slot,ix in d.groupby('slot',sort=False).indices.items():
        g=d.iloc[ix];order=np.argsort(-g[col].to_numpy(),kind='stable');y=g.ev.to_numpy()[order];den=min(int(g.m_p.iloc[0]),5)
        r.append({'slot':slot,'selected5_oracle':float(min(y[:5].sum(),5)/den),'top12_oracle':float(min(y[:12].sum(),5)/den),'top20_oracle':float(min(y[:20].sum(),5)/den)})
    return pd.DataFrame(r)

def run(input_root,output):
    R=Path(input_root);O=Path(output);O.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(R/'artifacts/oof_predictions.csv.gz');base=pair_metrics(d,d.base.to_numpy());cur=pair_metrics(d,d.candidate.to_numpy())
    b=d.base.to_numpy();v=d.candidate.to_numpy()-b
    groups=list(d.groupby('slot',sort=False).indices.values());top=lambda s,ix:set(ix[np.argsort(-s[ix],kind='stable')[:5]])
    rows=[]
    for alpha in [0.,.25,1.,2.,4.,8.,16.]:
        score=b+alpha*v;m=pair_metrics(d,score);bs=bootstrap(base,m)
        rows.append({'alpha':alpha,'E':float(m.E.mean()),'delta':bs['delta'],'wins':bs['wins'],'losses':bs['losses'],'top5_set_changes':sum(top(b,ix)!=top(score,ix) for ix in groups)})
    pd.DataFrame(rows).to_csv(O/'witness_residual_mobility.csv',index=False)
    geo=geometry(d,'base');geo.to_csv(O/'baseline_geometry.csv',index=False)
    checks={'witness_baseline_E':float(base.E.mean()),'witness_candidate_E':float(cur.E.mean()),'witness_paired':bootstrap(base,cur),'baseline_oracles':geo.drop(columns='slot').mean().to_dict(),'rows':len(d),'pairs':len(groups),'unique_pools':int(d.pool.nunique())}
    # Given the supplied ordering, the latent reversed get_indexer defect cancels.
    idx=pd.MultiIndex.from_frame(d[['slot','hand_id']]);sorted_d=d.sort_values(['slot','hand_id'],kind='stable');target=pd.MultiIndex.from_frame(sorted_d[['slot','hand_id']])
    wrong=idx.get_indexer(target);correct=target.get_indexer(idx)
    checks['supplied_order_reconstruction']={'wrong_equals_correct':bool(np.array_equal(wrong,correct)),'identity':bool(np.array_equal(wrong,np.arange(len(d)))),'note':'Uses supplied OOF order; original dev_pack/meta was not supplied, so full independent reconstruction of that source order is not possible.'}
    scores=pd.read_csv(R/'artifacts/rank_changes.csv.gz')
    def ap(y,s):
        j=np.argsort(-s,kind='stable');t=y[j];return float(np.sum(t*np.cumsum(t)/np.arange(1,len(t)+1))/sum(t))
    yy=(scores.label.to_numpy()==1).astype(int)
    checks['deduplicated_P_proxy']={'rows':len(scores),'positive':int(yy.sum()),'confirmed_negative':int((scores.label==0).sum()),'unknown':int((scores.label==-1).sum()),'base':ap(yy,scores.base_score.to_numpy()),'candidate':ap(yy,scores.new_score.to_numpy()),'limitation':'Original window/source labels were dropped; this is not a clean phase-specific validation.'}
    for c in ['base_score','new_score']:
        order=np.argsort(-scores[c].to_numpy(),kind='stable')
        checks['deduplicated_P_proxy'][c+'_top100_P']=int(yy[order[:100]].sum())
    paths=['scripts/round11/r11_entry.py','scripts/round12/r12_p_expected.py','scripts/round12/r12_p1.py','scripts/round13/r13_p_audit_prep.py','scripts/round13/r13_witness_diag.py','scripts/round13/decision_witness.py','scripts/round11/legacy_r8/replay.py','artifacts/pack.json']
    checks['source_sha256']={p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in paths}
    (O/'artifact_audit.json').write_text(json.dumps(checks,indent=2))
    return checks
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-root',required=True);p.add_argument('--output',required=True);print(json.dumps(run(**vars(p.parse_args())),indent=2))
