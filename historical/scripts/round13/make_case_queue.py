"""Export labeled hard swaps and per-pair losses for raw replay on GB10.
Uses dev labels for diagnosis only; never creates inference rules from these IDs.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from exact_map5 import ap_at_5

def run(path,out):
    d=pd.read_csv(path);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for slot,g in d.groupby('slot',sort=False):
        g=g.sort_values('rs_blend',ascending=False,kind='stable').reset_index(drop=True)
        y=g.ev.to_numpy(float);den=min(int(g.m_p.iloc[0]),5);base=ap_at_5(y,den)
        for a in range(min(5,len(g))):
            if y[a] != 0:continue
            for b in range(5,min(12,len(g))):
                if y[b] != 1:continue
                changed=y.copy();changed[a],changed[b]=changed[b],changed[a]
                rows.append(dict(slot=slot,pool=g.pool.iloc[0],fold=g.fold.iloc[0],
                    rejected_positive_hand_id=g.hand_id.iloc[b],selected_negative_hand_id=g.hand_id.iloc[a],
                    positive_rank=b+1,negative_rank=a+1,
                    positive_score=g.rs_blend.iloc[b],negative_score=g.rs_blend.iloc[a],
                    score_margin=g.rs_blend.iloc[a]-g.rs_blend.iloc[b],base_E=base,
                    oracle_single_swap_delta=float(ap_at_5(changed,den)-base),
                    full_truth_count=int(g.m_p.iloc[0])))
    cases=pd.DataFrame(rows).sort_values(['oracle_single_swap_delta','score_margin'],ascending=[False,True],kind='stable')
    cases.to_csv(out/'all_top12_hard_swaps.csv',index=False)
    # One case per pair avoids spending every inspection on the same failure.
    cases.drop_duplicates('slot').head(60).to_csv(out/'priority_60_distinct_pair_cases.csv',index=False)
    report={'hard_swaps':len(cases),'distinct_pairs':int(cases.slot.nunique()),
            'scope':'Labeled OOF error diagnostic; no original actions in this package. Reviewer must replay these hands on GB10.',
            'non_additive':'Single-swap oracle deltas overlap and MUST NOT be summed as achievable gain.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.input,a.out)
