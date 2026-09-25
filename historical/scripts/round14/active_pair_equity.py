"""Enable current-street investigator HU equity without changing actor policy.

Six players' actual private cards are retrospective operator information. Future
board cards are NOT used before they appear. River/turn are exact; the flop can
use a fixed number of reproducible samples from legal runouts. No general action
EV claim: the legacy witness's guarded river-call EV remains narrowly scoped.
"""
from __future__ import annotations
import hashlib,itertools
import numpy as np

def attach_pair_equity(records:list[dict],holes:np.ndarray,board:np.ndarray,A:int,B:int,engine,mode:str='postflop',flop_samples:int=96) -> list[dict]:
    if mode not in ('none','river','postflop','exact'):raise ValueError('Unknown equity mode')
    if mode=='none':return [dict(r) for r in records]
    holes=np.asarray(holes,dtype=np.int16);board=np.asarray(board,dtype=np.int16);board=board[board>=0]
    if holes.shape!=(6,2) or A==B or not (0<=A<6 and 0<=B<6):raise ValueError('Six seats and two different roles required')
    if flop_samples<1:raise ValueError('Positive sample count required')
    cache={};out=[]
    for r in records:
        st=int(r['st']);z=dict(r);z['hu']=np.asarray(r['hu'],dtype=float).copy()
        if st<1 or (mode=='river' and st!=3):out.append(z);continue
        prefix=board[:st+2]
        if len(prefix)!=st+2:raise ValueError('Missing observed board prefix')
        if st not in cache:
            known=np.r_[holes.ravel(),prefix]
            if len(np.unique(known))!=len(known):raise ValueError('Duplicate known cards')
            available=np.setdiff1d(np.arange(52,dtype=np.int16),known);need=5-len(prefix)
            cs=np.array(list(itertools.combinations(available,need)),dtype=np.int16).reshape(-1,need) if need else np.zeros((1,0),np.int16)
            exact=True
            if st==1 and mode!='exact' and len(cs)>flop_samples:
                # Seed from actual cards/prefix, not IDs or file order.
                seed=int.from_bytes(hashlib.sha256(known.tobytes()).digest()[:8],'little')
                cs=cs[np.random.default_rng(seed).choice(len(cs),flop_samples,replace=False)];exact=False
            vals=np.asarray([[engine.made_value(np.r_[holes[s],prefix,c]) for s in (A,B)] for c in cs])
            q=float(np.mean(vals[:,0]>vals[:,1])+.5*np.mean(vals[:,0]==vals[:,1]))
            cache[st]=(q,exact,len(cs))
        q,exact,n=cache[st];z['hu'][A,B]=q;z['hu'][B,A]=1-q
        z['pair_equity_exact']=exact;z['pair_equity_runouts']=n
        out.append(z)
    return out
