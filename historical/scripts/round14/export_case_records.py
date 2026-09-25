"""GB10 raw development-case replay and activated witness pack, no submission.
Export JSONL records so actual mechanism review does not stop at a tensor build.
"""
from __future__ import annotations
import argparse,importlib,json,sys,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from active_pair_equity import attach_pair_equity

def convert(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)

def run(candidates,selected_slots,np_dir,legacy_code,witness_code,out,equity_mode='postflop',flop_samples=96):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new output directory')
    out.mkdir(parents=True,exist_ok=True);t=time.monotonic()
    sys.path[:0]=[str(Path(legacy_code).resolve()),str(Path(witness_code).resolve())]
    replay=importlib.import_module('replay');engine=importlib.import_module('equity');wit=importlib.import_module('decision_witness')
    if Path(replay.__file__).resolve().parent!=Path(legacy_code).resolve():raise RuntimeError('Wrong replay import')
    a=replay.Arrays(np_dir);d=pd.read_csv(candidates)
    if selected_slots:d=d[d.slot.isin(pd.read_csv(selected_slots).slot)]
    d=d.sort_values(['slot','hand_id'],kind='stable').reset_index(drop=True)
    if not len(d) or d.duplicated(['slot','hand_id']).any():raise ValueError('Empty/duplicate candidates')
    hi=pd.read_parquet(Path(np_dir)/'hand_index.parquet').set_index('hand_id').hi
    pi=pd.read_parquet(Path(np_dir)/'player_index.parquet').set_index('player_id').pi
    dim=len(wit.feature_names());x=np.lib.format.open_memmap(out/'witness_views.npy',mode='w+',dtype='float32',shape=(len(d),2,dim))
    selected=['st','i','act','y','aggr','amt','tc','pot','stack','bb','la','raise_proxy','alive','can','total','probs','prob_known','geom','hu','eq96','eqla96','faced','pair_equity_exact','pair_equity_runouts']
    check={'max_stack_error':0.,'illegal_actors':0,'postflop_pair_records':0,'active_pair_records':0}
    with (out/'raw_cases.jsonl').open('w') as f:
        for j,row in enumerate(d.itertuples(index=False)):
            h=int(hi.loc[row.hand_id]);pa=int(pi.loc[row.pair_player_lo]);pb=int(pi.loc[row.pair_player_hi])
            if int(a['h_phase'][h])!=0:raise ValueError('Only development-case inspection is allowed by this exporter')
            ia=np.flatnonzero(a['s_player'][h]==pa);ib=np.flatnonzero(a['s_player'][h]==pb)
            if len(ia)!=1 or len(ib)!=1:raise ValueError('Pair not seated')
            A,B=int(ia[0]),int(ib[0]);records,holes,board,c=replay.replay_hand(h,a,exact=False)
            records=attach_pair_equity(records,holes,board,A,B,engine,mode=equity_mode,flop_samples=flop_samples)
            check['max_stack_error']=max(check['max_stack_error'],c['max_stack_error']);check['illegal_actors']+=c['illegal_actors']
            for r in records:
                if r['i'] in (A,B) and r['st']>0:
                    check['postflop_pair_records']+=1
                    other=B if r['i']==A else A
                    check['active_pair_records']+=int(0<=r['hu'][r['i'],other]<=1)
            x[j]=wit.two_views(records,A,B)
            f.write(json.dumps({'slot':row.slot,'hand_id':row.hand_id,'reference_label':getattr(row,'ev',None),'seats':[A,B],'holes':holes,'board':board,'records':[{k:r[k] for k in selected if k in r} for r in records]},default=convert)+'\n')
            if (j+1)%100==0:print(f'{j+1}/{len(d)}',flush=True)
    x.flush();d.to_csv(out/'meta.csv',index=False);(out/'feature_names.json').write_text(json.dumps(wit.feature_names()))
    receipt={'rows':len(d),'pairs':int(d.slot.nunique()),'equity_mode':equity_mode,'flop_samples':flop_samples,'checks':check,'seconds':time.monotonic()-t,'input_sha256':hashlib.sha256(Path(candidates).read_bytes()).hexdigest(),'state':'extracted_only_not_trained'}
    (out/'activation_receipt.json').write_text(json.dumps(receipt,indent=2));return receipt
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['candidates','np-dir','legacy-code','witness-code','out']:p.add_argument('--'+n,required=True)
    p.add_argument('--selected-slots');p.add_argument('--equity-mode',choices=['none','river','postflop','exact'],default='postflop');p.add_argument('--flop-samples',type=int,default=96)
    print(json.dumps(run(**vars(p.parse_args())),indent=2))
