"""Independent card evaluator and exact postflop runouts. No generator internals.
All six hole cards are known to the investigator and removed from the deck,
including folded players' cards. Returns showdown shares, not general action EV.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np
from numba import njit

@njit(cache=True)
def _straight(mask):
    for hi in range(14,5-1,-1):
        if ((mask>>(hi-4))&31)==31:return hi
    if mask & ((1<<14)|(1<<5)|(1<<4)|(1<<3)|(1<<2)) == ((1<<14)|(1<<5)|(1<<4)|(1<<3)|(1<<2)):return 5
    return 0

@njit(cache=True)
def made_value(cards):
    cnt=np.zeros(15,np.int64);sc=np.zeros(4,np.int64);sm=np.zeros(4,np.int64);mask=0
    for card in cards:
        if card<0:continue
        r=card//4+2;s=card%4;cnt[r]+=1;sc[s]+=1;sm[s]|=1<<r;mask|=1<<r
    B=15**5
    fs=-1
    for s in range(4):
        if sc[s]>=5:fs=s
    if fs>=0:
        v=_straight(sm[fs])
        if v:return 8*B+v*15**4
    qu=0;tr=0;pa=0
    for r in range(14,1,-1):
        if cnt[r]==4 and not qu:qu=r
        if cnt[r]>=3 and not tr:tr=r
    if qu:
        for r in range(14,1,-1):
            if r!=qu and cnt[r]:return 7*B+qu*15**4+r*15**3
    if tr:
        for r in range(14,1,-1):
            if r!=tr and cnt[r]>=2:pa=r;break
        if pa:return 6*B+tr*15**4+pa*15**3
    if fs>=0:
        v=5*B;k=4
        for r in range(14,1,-1):
            if sm[fs]&(1<<r):v+=r*15**k;k-=1
            if k<0:break
        return v
    st=_straight(mask)
    if st:return 4*B+st*15**4
    if tr:
        v=3*B+tr*15**4;k=3
        for r in range(14,1,-1):
            if r!=tr and cnt[r]:v+=r*15**k;k-=1
            if k==1:break
        return v
    pairs=[]
    for r in range(14,1,-1):
        if cnt[r]>=2:pairs.append(r)
    if len(pairs)>=2:
        a,b=pairs[0],pairs[1]
        for r in range(14,1,-1):
            if r!=a and r!=b and cnt[r]:return 2*B+a*15**4+b*15**3+r*15**2
    if len(pairs)==1:
        a=pairs[0];v=B+a*15**4;k=3
        for r in range(14,1,-1):
            if r!=a and cnt[r]:v+=r*15**k;k-=1
            if k==0:break
        return v
    v=0;k=4
    for r in range(14,1,-1):
        if cnt[r]:v+=r*15**k;k-=1
        if k<0:break
    return v

@njit(cache=True)
def _score_runouts(holes, board, completions):
    out=np.empty((len(completions),6),np.int64)
    cards=np.empty(7,np.int16)
    for k in range(len(completions)):
        for s in range(6):
            cards[0]=holes[s,0];cards[1]=holes[s,1]
            for b in range(len(board)):cards[2+b]=board[b]
            for j in range(completions.shape[1]):cards[2+len(board)+j]=completions[k,j]
            out[k,s]=made_value(cards)
    return out

def runout_scores(holes: np.ndarray, board_prefix: np.ndarray):
    holes=np.asarray(holes,np.int16);board=np.asarray(board_prefix,np.int16)
    if holes.shape!=(6,2) or len(board) not in (3,4,5):raise ValueError('Six holes and flop/turn/river prefix required')
    known=np.r_[holes.ravel(),board]
    if len(set(known.tolist()))!=len(known) or np.any((known<0)|(known>=52)):raise ValueError('Duplicate/invalid known cards')
    available=sorted(set(range(52))-set(known.tolist()));need=5-len(board)
    cs=np.array(list(combinations(available,need)),np.int16).reshape(-1,need) if need else np.zeros((1,0),np.int16)
    return _score_runouts(holes,board,cs)

def shares(scores: np.ndarray, eligible: np.ndarray):
    eligible=np.asarray(eligible,bool)
    if eligible.shape!=(6,) or not eligible.any():raise ValueError('Nonempty eligibility mask required')
    best=scores[:,eligible].max(1);win=(scores==best[:,None])&eligible[None,:]
    return (win/win.sum(1,keepdims=True)).mean(0)

def pot_layers(contributions: np.ndarray, live: np.ndarray):
    """Current committed-chip layers, including folded dead money.
    Uncontested single-contributor excess is returned explicitly; it is not a pot.
    """
    c=np.asarray(contributions,float);live=np.asarray(live,bool)
    if c.shape!=(6,) or live.shape!=(6,) or (c<0).any():raise ValueError('Six nonnegative contributions required')
    old=0.;layers=[];refund=np.zeros(6)
    for level in np.unique(c[c>0]):
        covered=c>=level;amount=(level-old)*covered.sum();old=level
        if covered.sum()==1:refund[covered]+=amount;continue
        elig=covered&live
        if not elig.any():raise ValueError('No eligible live player for committed pot layer')
        layers.append((float(amount),elig))
    return layers,refund

def expected_award(scores,contributions,live):
    layers,refund=pot_layers(contributions,live);award=refund.copy()
    for amount,eligible in layers:award+=amount*shares(scores,eligible)
    return award
