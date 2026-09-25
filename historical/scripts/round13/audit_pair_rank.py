"""Diagnose risk-order changes without mistaking unlabeled pairs for negatives.

Input CSV: pair_id,pool,label,base_score,new_score; label is 1/0/-1 (unknown).
Optional q_unknown is an externally supplied SENSITIVITY SCENARIO, not truth.
No submission, no model training, no model-dependent row deletion.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def ordered_ap(y: np.ndarray, scores: np.ndarray):
    y = np.asarray(y, int); scores = np.asarray(scores, float)
    order = np.argsort(-scores, kind='stable')
    ranks = np.empty(len(y), int); ranks[order] = np.arange(1, len(y) + 1)
    contribution = np.zeros(len(y), float)
    if y.sum(): contribution[order] = y[order] * np.cumsum(y[order]) / np.arange(1, len(y)+1) / y.sum()
    return float(contribution.sum()), ranks, contribution


def run(path, out, scenario_draws=100, seed=20260918):
    d = pd.read_csv(path).sort_values('pair_id', kind='stable').reset_index(drop=True)
    required = {'pair_id','pool','label','base_score','new_score'}
    if required - set(d): raise ValueError(f'Missing columns {required - set(d)}')
    if d.pair_id.duplicated().any(): raise ValueError('Duplicate pairs')
    if not d.label.isin([-1,0,1]).all(): raise ValueError('Label must be -1,0,1')
    if not np.isfinite(d[['base_score','new_score']]).all().all(): raise ValueError('Invalid scores')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    yy = (d.label == 1).to_numpy(int)
    ab, rb, cb = ordered_ap(yy, d.base_score.to_numpy())
    an, rn, cn = ordered_ap(yy, d.new_score.to_numpy())
    d['base_rank'] = rb; d['new_rank'] = rn; d['rank_improvement'] = rb - rn
    d['proxy_ap_delta_contribution'] = cn - cb
    cuts = {}
    for k in (100,300,450,1000,2000,4000,8000,len(d)):
        if k > len(d): continue
        b, n = rb <= k, rn <= k
        cuts[str(k)] = dict(shared_ids=int((b&n).sum()), same_membership=bool(np.array_equal(b,n)),
                           known_pos_base=int(yy[b].sum()), known_pos_new=int(yy[n].sum()),
                           unknown_base=int((d.label.to_numpy()[b] == -1).sum()),
                           unknown_new=int((d.label.to_numpy()[n] == -1).sum()),
                           base_ap_prefix_contribution=float(cb[b].sum()),
                           new_ap_prefix_contribution=float(cn[n].sum()))
    report = {'scope':'AP here is a KNOWN-POSITIVE RETRIEVAL PROXY; unknowns are not verified negatives.',
              'known_positive_proxy':{'base':ab,'new':an,'delta':an-ab},'top_regions':cuts,
              'unknown_count':int((d.label==-1).sum()),'scenario_draws':0}
    if 'q_unknown' in d:
        q = d.q_unknown.to_numpy(float)
        if not np.isfinite(q).all() or np.any((q<0)|(q>1)): raise ValueError('Invalid scenario probabilities')
        q = np.where(d.label == 1,1.,np.where(d.label == 0,0.,q))
        rng=np.random.default_rng(seed);delta=[]
        for _ in range(scenario_draws):
            y = (rng.random(len(q)) < q).astype(int)
            delta.append(ordered_ap(y,d.new_score.to_numpy())[0] - ordered_ap(y,d.base_score.to_numpy())[0])
        report['unknown_label_sensitivity']={'delta_mean':float(np.mean(delta)),
            'delta_percentiles_2_5_50_97_5':np.quantile(delta,[.025,.5,.975]).tolist(),
            'NOT_a_confidence_interval':True,
            'assumption':'Independent Bernoulli labels at supplied q_unknown; requires separate correlated scenarios.'}
        report['scenario_draws']=scenario_draws
    d.to_csv(out/'rank_changes.csv.gz',index=False)
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    p.add_argument('--scenario-draws',type=int,default=100);p.add_argument('--seed',type=int,default=20260918)
    a=p.parse_args();run(a.input,a.out,a.scenario_draws,a.seed)
