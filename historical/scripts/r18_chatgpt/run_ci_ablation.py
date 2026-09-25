#!/usr/bin/env python3
"""Reproduce the R18 CI feature/censoring ablations on existing R3 tables."""
import argparse,json
from pathlib import Path
import numpy as np
import ci_censored_event as ci
p=argparse.ArgumentParser();p.add_argument('--full',required=True);p.add_argument('--candidates',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=260919);a=p.parse_args()
full=ci.read_frame(a.full);cand=ci.read_frame(a.candidates)
original=list(ci.FEATURES);censor=ci.uncensored_training_rows
response={'o_fold_after','o_callraise_after','end_pre','o_alive_endpre','out_won','out_sd','all_out_folded','both_flop','o_fold_fraction'}
cards={'own','par','card_gap'}
outcomes={'actor_net','partner_net','actor_win','partner_win','actor_fold','partner_fold','pair_net','sd_any','iso','pair_win','xfer'}
rows=[]
for mode,drop in [('full',set()),('no_response',response),('no_cards',cards),('no_outcomes',outcomes),('no_censor',set())]:
    ci.FEATURES=[f for f in original if f not in drop]
    ci.uncensored_training_rows=(lambda s:np.ones(len(s),dtype=bool)) if mode=='no_censor' else censor
    summary,_,_=ci.evaluate(full,cand,seed=a.seed);summary['mode']=mode;rows.append(summary);print(json.dumps(summary),flush=True)
ci.FEATURES=original;ci.uncensored_training_rows=censor
Path(a.out).write_text(json.dumps(rows,indent=2))
