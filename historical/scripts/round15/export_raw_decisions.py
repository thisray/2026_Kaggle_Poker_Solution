"""GB10 exporter: reuse legacy replay without rebuilding 5904-column witnesses.
Inference data can be exported for model prediction; never hand-label evaluation.
"""
from __future__ import annotations
import argparse, importlib, json, sys
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from action_mil import encode_record

def conv(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)

def setup(path,legacy):
    sys.path.insert(0,str(Path(legacy).resolve()));replay=importlib.import_module('replay')
    if Path(replay.__file__).resolve().parent!=Path(legacy).resolve():raise RuntimeError('Wrong replay module')
    return replay,replay.Arrays(path)

def main(a):
    replay,arr=setup(a.np_dir,a.legacy_code);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    hi=pd.read_parquet(Path(a.np_dir)/'hand_index.parquet')
    if a.mode=='candidates':
        d=pd.read_csv(a.candidates);hp=hi.set_index('hand_id').hi
        pi=pd.read_parquet(Path(a.np_dir)/'player_index.parquet').set_index('player_id').pi
        @lru_cache(maxsize=512)
        def get(h):return replay.replay_hand(h,arr,exact=a.exact)
        n=0
        with (out/'raw_decisions.jsonl').open('w') as f:
            for row in d.itertuples(index=False):
                h=int(hp.loc[row.hand_id]);players=arr['s_player'][h]
                A=np.flatnonzero(players==int(pi.loc[row.pair_player_lo]));B=np.flatnonzero(players==int(pi.loc[row.pair_player_hi]))
                if len(A)!=1 or len(B)!=1:raise ValueError('Pair membership')
                records,holes,board,checks=get(h)
                f.write(json.dumps({'slot':row.slot,'hand_id':row.hand_id,'seats':[int(A[0]),int(B[0])],
                    'phase':int(arr['h_phase'][h]),'holes':holes,'board':board,'records':records},default=conv)+'\n');n+=1
        d.to_csv(out/'meta.csv',index=False)
        print(json.dumps({'rows':n,'exact':a.exact,'output':'raw_decisions.jsonl','no_training_or_submission':True}))
    else:
        # Supply the existing hand->pool fold manifest. Do not derive pools from ID formats.
        m=pd.read_csv(a.hand_manifest)
        required={'hand_id','pool','fold','phase'}
        if not required.issubset(m):raise ValueError(f'Need {required}')
        if m.hand_id.duplicated().any() or m.groupby('pool').fold.nunique().max()!=1:raise ValueError('Manifest conflict')
        m=m[m.phase.astype(str).isin(['0','development'])].merge(hi[['hand_id','hi']],on='hand_id',validate='one_to_one')
        rng=np.random.default_rng(a.seed);parts=[]
        # Balanced pool sampling supports unseen-pool policy validation.
        quota=max(1,a.hands//m.pool.nunique())
        for pool,g in m.groupby('pool'):
            parts.append(g.iloc[rng.choice(len(g),min(quota,len(g)),replace=False)])
        m=pd.concat(parts,ignore_index=True);ctx=[];act=[];fold=[];pool=[]
        for row in m.itertuples(index=False):
            records,_,_,_=replay.replay_hand(int(row.hi),arr,exact=False)
            for k,r in enumerate(records):
                c,_,y=encode_record(r,0,1,k,len(records));ctx.append(c);act.append(y);fold.append(row.fold);pool.append(row.pool)
        np.savez_compressed(out/'background.npz',context=np.asarray(ctx,np.float32),action=np.asarray(act,np.int64),
             fold=np.asarray(fold),pool=np.asarray(pool))
        print(json.dumps({'hands':len(m),'actions':len(act),'no_label_information':True}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['candidates','background'])
    for k in ['np-dir','legacy-code','out']:p.add_argument('--'+k,required=True)
    p.add_argument('--candidates');p.add_argument('--hand-manifest');p.add_argument('--hands',type=int,default=20000)
    p.add_argument('--seed',type=int,default=20260918);p.add_argument('--exact',action='store_true');main(p.parse_args())
