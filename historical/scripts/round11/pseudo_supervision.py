"""Abstaining, mass-capped pseudo supervision, not automatic top-five truth.
Input probabilities must come from outer-pool-excluded, inner-calibrated teachers.
This module only prepares targets/weights and pairwise preferences. It does not
establish calibration, pair precision or competition gain for the caller.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def prepare(d: pd.DataFrame, heldout_pools=(), high=.85, low=.05, pair_floor=.9,
            query_mass=.25, human_total_weight=372., total_ratio=.5):
    req={'slot','pool','hand_id','q_pair','q_low','q_high'}
    if req-set(d): raise ValueError(f'Missing {req-set(d)}')
    if not 0 <= low < high <= 1 or query_mass<=0 or human_total_weight<=0 or total_ratio<=0:
        raise ValueError('Invalid thresholds or weight budgets')
    if d.duplicated(['slot','hand_id']).any():raise ValueError('Duplicate pseudo keys')
    if set(d.pool)&set(heldout_pools):raise ValueError('Outer-held-out pool included in pseudo training')
    q=d[['q_pair','q_low','q_high']].to_numpy(float)
    if not np.isfinite(q).all() or np.any((q<0)|(q>1)) or np.any(q[:,1]>q[:,2]):
        raise ValueError('Require bounded teacher confidence interval estimates')
    # All thresholds/intervals must be chosen using inner training labels, never LB.
    r=d.copy().reset_index(drop=True)
    r['soft_target']=(r.q_low+r.q_high)/2
    positive=(r.q_low>=high)&(r.q_pair>=pair_floor)
    negative=(r.q_high<=low)&(r.q_pair>=pair_floor)
    r['kind']=np.where(positive,'positive',np.where(negative,'negative','abstain'))
    confidence=np.where(positive,r.q_low,np.where(negative,1-r.q_high,0.))
    r['weight']=np.asarray(confidence)*r.q_pair*(1-(r.q_high-r.q_low))
    for ix in r.groupby('slot',sort=False).indices.values():
        mass=float(r.loc[ix,'weight'].sum())
        if mass>0:r.loc[ix,'weight']*=query_mass/mass
    total=float(r.weight.sum());cap=human_total_weight*total_ratio
    if total>cap:r.weight*=cap/total
    receipt={'rows':len(r),'selected_rows':int((r.weight>0).sum()),'positive':int(positive.sum()),
             'negative':int(negative.sum()),'abstain':int((r.kind=='abstain').sum()),
             'total_weight':float(r.weight.sum()),'max_query_weight':float(r.groupby('slot').weight.sum().max()) if len(r) else 0.,
             'assumption':'intervals and q_pair calibrated in inner pools; no accuracy guarantee provided'}
    return r,receipt


def preference_pairs(d: pd.DataFrame, min_gap=.3, max_pairs_per_query=25):
    """Accept only non-overlapping teacher intervals. Unselected rows are unknown.
    These are weighted pairwise preferences, not integer LambdaRank labels.
    """
    rows=[]
    for slot,g in d.groupby('slot',sort=False):
        prefs=[]
        for i,a in g.iterrows():
            for j,b in g.iterrows():
                if i==j:continue
                gap=float(a.q_low-b.q_high)
                if gap>=min_gap:
                    prefs.append((gap,str(a.hand_id),str(b.hand_id)))
        prefs=sorted(prefs,reverse=True)[:max_pairs_per_query]
        mass=sum(x[0] for x in prefs)
        budget=float(g.weight.sum())
        for gap,a,b in prefs:
            rows.append({'slot':slot,'preferred_hand_id':a,'other_hand_id':b,
                         'weight':budget*gap/mass if mass else 0.})
    return pd.DataFrame(rows,columns=['slot','preferred_hand_id','other_hand_id','weight'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    p.add_argument('--heldout-pools',default='');p.add_argument('--human-total-weight',type=float,required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    held=[int(x) for x in a.heldout_pools.split(',') if x]
    d,r=prepare(pd.read_csv(a.input),heldout_pools=held,human_total_weight=a.human_total_weight)
    d.to_csv(out/'soft_supervision.csv',index=False);preference_pairs(d).to_csv(out/'preferences.csv',index=False)
    (out/'receipt.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
