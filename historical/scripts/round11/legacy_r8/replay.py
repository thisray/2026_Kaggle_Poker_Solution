"""Decision-local records over existing opus np exports. IDs are never features.
Uses chip increments, not guessed amount_to semantics. Reopen is a documented
NLHE opportunity proxy; the competition's exact reopening convention is unverified.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from equity import made_value,runout_scores,shares

REQUIRED=['a_off','a_st','a_seat','a_act','a_amount','a_to_call','a_pot_before',
          'a_stack_before','h_bb','h_sb','h_btn','h_board','s_player','s_stack','s_c1','s_c2','h_phase']
class Arrays:
    def __init__(self,path):
        root=Path(path);self.d={}
        for n in REQUIRED:
            p=root/f'{n}.npy'
            if not p.exists():raise FileNotFoundError(f'{p}: run existing scripts/opus_r1/e1_export.py first')
            self.d[n]=np.load(p,mmap_mode='r',allow_pickle=False)
        for name,file in [('probs','dec_probs_v1.npy'),('eqm96','act_eqm_v1.npy'),('eqla96','act_eqla_v1.npy')]:
            p=root.parent/file
            self.d[name]=np.load(p,mmap_mode='r',allow_pickle=False) if p.exists() else None
    def __getitem__(self,key):return self.d[key]

def card_geometry(hole,board):
    allc=np.r_[hole,board];r=allc//4+2;s=allc%4;hr=hole//4+2
    cnt=np.bincount(s,minlength=4)
    mask=set(r.tolist());mask|=({1} if 14 in mask else set())
    straight_near=max(sum(x in mask for x in range(start,start+5)) for start in range(1,11))
    v=made_value(allc) if len(board)>=3 else -1
    return np.array([v//15**5 if v>=0 else -1,max(hr)/14,min(hr)/14,
                     float(hr[0]==hr[1]),float(hole[0]%4==hole[1]%4),abs(int(hr[0])-int(hr[1]))/12,
                     float(cnt.max()==4 and len(board)<5),float(cnt.max()>=5),straight_near/5,
                     float(len(set((board//4).tolist()))<len(board))],np.float32)

def replay_hand(h:int,a:Arrays,exact:bool=False):
    holes=np.column_stack([a['s_c1'][h],a['s_c2'][h]]).astype(np.int16)
    board=np.asarray(a['h_board'][h],np.int16);board=board[board>=0]
    known=np.r_[holes.ravel(),board]
    if len(set(known.tolist()))!=len(known):raise ValueError(f'duplicate cards at h={h}')
    bb=float(a['h_bb'][h]);
    if bb<=0:raise ValueError('Positive big blind required')
    btn=int(a['h_btn'][h]);total=np.zeros(6);bets=np.zeros(6)
    stacks=np.asarray(a['s_stack'][h],float).copy();alive=np.ones(6,bool)
    for seat,amount in [((btn+1)%6,float(a['h_sb'][h])),((btn+2)%6,bb)]:
        q=min(stacks[seat],amount);total[seat]=bets[seat]=q;stacks[seat]-=q
    acted=np.full(6,np.nan);cur=-1;la=-1;last_full=bb;raises=0
    # For each responder retain the distinct players whose aggression they actually faced.
    faced=np.zeros(6,np.int64);records=[];eqcache={};geomcache={};hucache={};max_stack_error=0.;illegal_actors=0
    for k in range(int(a['a_off'][h]),int(a['a_off'][h+1])):
        st=int(a['a_st'][k]);i=int(a['a_seat'][k]);act=int(a['a_act'][k])
        if st!=cur:
            if cur>=0:bets[:]=0
            cur=st;la=-1;last_full=bb;raises=0;acted[:]=np.nan;faced[:]=0
        amt=float(a['a_amount'][k]);tc=float(a['a_to_call'][k]);pot=float(a['a_pot_before'][k]);reported=float(a['a_stack_before'][k])
        max_stack_error=max(max_stack_error,abs(stacks[i]-reported));stacks[i]=reported
        if not alive[i] or reported<amt-1e-5:illegal_actors+=1
        can=alive&(stacks>1e-6);current=float(bets.max())
        # A check before an opening wager retains raising opportunity. Other reopen
        # decisions are player-specific, including cumulative short raises.
        reopen=np.isnan(acted[i]) or current-acted[i]>=last_full-1e-6
        raise_proxy=float(can[i] and reported>tc+1e-6 and np.any(can&(np.arange(6)!=i)) and reopen)
        aggr=act in (3,4) or (act==5 and amt>tc+1e-6)
        action4=0 if act==0 else (1 if act==1 else (3 if aggr else 2))
        prefix=board[:0 if st==0 else 2+st]
        if st not in geomcache:geomcache[st]=np.vstack([card_geometry(holes[s],prefix) for s in range(6)])
        geom=geomcache[st]
        full=np.full(6,-1.);hu=np.full((6,6),-1.)
        if exact and len(prefix)>=3:
            if st not in eqcache:eqcache[st]=runout_scores(holes,prefix)
            scores=eqcache[st];full=shares(scores,alive)
            if st not in hucache:
                for u in range(6):
                    for v in range(u+1,6):
                        win=(scores[:,u]>scores[:,v]).mean()+.5*(scores[:,u]==scores[:,v]).mean();hu[u,v]=win;hu[v,u]=1-win
                hucache[st]=hu
            hu=hucache[st]
        probs=np.asarray(a['probs'][k],float) if a['probs'] is not None else np.ones(4)/4
        eq96=float(a['eqm96'][k]) if a['eqm96'] is not None else -1.
        eqla96=float(a['eqla96'][k]) if a['eqla96'] is not None else -1.
        if tc>0 and la>=0:faced[i]|=1<<la
        records.append(dict(st=st,i=i,act=act,y=action4,aggr=aggr,amt=amt,tc=tc,pot=pot,
            stack=reported,bb=bb,btn=btn,la=la,raises=raises,last_full=last_full,raise_proxy=raise_proxy,
            alive=alive.copy(),can=can.copy(),stacks=stacks.copy(),total=total.copy(),bets=bets.copy(),
            faced=faced.copy(),geom=geom,full=full,hu=hu,probs=probs,prob_known=a['probs'] is not None,
            eq96=eq96,eqla96=eqla96))
        if aggr:
            increment=bets[i]+amt-current
            if increment>=last_full-1e-6:last_full=increment
            la=i;raises+=1
        if act==0:alive[i]=False
        total[i]+=amt;bets[i]+=amt;stacks[i]=max(0.,reported-amt)
        if not (act==1 and current<=1e-6):acted[i]=float(bets.max())
    return records,holes,board,dict(max_stack_error=max_stack_error,illegal_actors=illegal_actors,
                                   action_count=len(records))

EVENTS=['pair_fold_partner','pair_call_partner','pair_check_hu','pair_aggr_outs',
        'outsider_faced_both_fold','pair_concession_after_same_outsider']
MOMENT_NAMES=['exact_eq','hu_eq','eq96','eq_change','terminal_call_ev_bb','pot_odds',
              'amount_pot','raise_opportunity_proxy','legal_aggr_mass','same_target_count',
              'actor_category','actor_hi','actor_lo','actor_pair','actor_suited','actor_gap',
              'actor_flush_draw','actor_flush_made','actor_straight_near','board_paired',
              'partner_category','partner_flush_draw','category_gap','eq_times_fold_price']
NUMERIC_NAMES=['log_pot','log_to_call','log_amount','log_stack','pot_odds','amount_pot',
 'live_n','can_act_n','facing_partner','outs_live_n','outs_can_n','partner_live','partner_can',
 'raise_opportunity_proxy','raise_count','log_last_full_raise','log_actor_total','actor_position',
 'p_fold','p_check','p_call','p_aggr','prob_known','eq96','eqla96','same_outsider_count']
for kind in ['remain','live','can','committed','street_commit']:NUMERIC_NAMES +=[f'{kind}_role{s}' for s in range(6)]
TAB_NAMES=[f'{ev}_{stat}_{name}' for ev in EVENTS for stat in ['mean','max'] for name in MOMENT_NAMES]+[f'{ev}_n' for ev in EVENTS]

def view(records,holes,board,A:int,B:int):
    outsiders=[s for s in range(6) if s not in (A,B)];outsiders.sort(key=lambda s:(s-records[0]['btn'])%6)
    seats=[A,B]+outsiders;inv={s:j for j,s in enumerate(seats)}
    cards=np.full((4,17),-1,np.int16)
    for st in range(4):
        cards[st,:12]=holes[seats].ravel();n=min(len(board),0 if st==0 else st+2);cards[st,12:12+n]=board[:n]
    values=[];cat=[];events=[[] for _ in EVENTS];exit_same=0
    for t,r in enumerate(records):
        i=r['i'];j=B if i==A else A if i==B else -1;pair=j>=0;bb=r['bb'];is_toward=pair and r['la']==j and r['tc']>0
        bothmask=(1<<A)|(1<<B)
        same=sum((int(r['faced'][o])&bothmask)==bothmask for o in outsiders)
        nouts=int(r['alive'][outsiders].sum());ncouts=int(r['can'][outsiders].sum())
        # Only observed responses count toward the shared-outsider chain.
        fold_both=not pair and r['act']==0 and r['la'] in (A,B) and (int(r['faced'][i])&bothmask)==bothmask
        if fold_both:exit_same+=1
        pj=bool(r['alive'][j]) if pair else False;cj=bool(r['can'][j]) if pair else False
        vals=[np.log1p(r['pot']/bb),np.log1p(r['tc']/bb),np.log1p(r['amt']/bb),np.log1p(r['stack']/bb),
              r['tc']/max(1e-8,r['pot']+r['tc']),r['amt']/max(bb,r['pot']),r['alive'].mean(),r['can'].mean(),
              float(is_toward),nouts/4,ncouts/4,float(pj),float(cj),r['raise_proxy'],r['raises']/10,
              np.log1p(r['last_full']/bb),np.log1p(r['total'][i]/bb),(i-r['btn'])%6/5,
              *r['probs'],float(r['prob_known']),r['eq96'],r['eqla96'],same/4]
        vals +=list(np.log1p(r['stacks'][seats]/bb))+list(r['alive'][seats])+list(r['can'][seats])
        vals +=list(np.log1p(r['total'][seats]/bb))+list(np.log1p(r['bets'][seats]/bb))
        values.append(vals);cat.append([inv[i],r['st'],r['act'],inv.get(r['la'],6)])
        q=float(r['full'][i]);qh=float(r['hu'][i,j]) if pair else -1.;g=r['geom'][i];pg=r['geom'][j] if pair else np.full(10,-1.)
        # Exact call-minus-fold EV is confined to a full-call river HU confrontation.
        post_total=r['total'].copy();post_total[i]+=r['tc']
        simple_pot=(pair and abs(post_total[i]-post_total[j])<1e-6 and all(post_total[o]<=post_total[i]+1e-6 for o in outsiders))
        terminal=(simple_pot and is_toward and r['st']==3 and r['alive'].sum()==2 and r['stack']>=r['tc'] and qh>=0)
        ev=(qh*(r['pot']+r['tc'])-r['tc'])/bb if terminal else 0.
        mm=[q,qh,r['eq96'],q-r['eq96'] if q>=0 and r['eq96']>=0 else 0.,ev,
            r['tc']/max(1e-8,r['pot']+r['tc']),r['amt']/max(bb,r['pot']),r['raise_proxy'],
            r['probs'][3]*r['raise_proxy'],float(same),*g,pg[0],pg[6],g[0]-pg[0],max(qh,0)*r['tc']/bb]
        masks=[pair and is_toward and r['act']==0,pair and is_toward and r['y']==2,
               pair and r['act']==1 and pj and r['alive'].sum()==2,
               pair and r['aggr'] and ncouts>0,fold_both,
               pair and is_toward and r['y'] in (0,2) and exit_same>0]
        for e,on in enumerate(masks):
            if on:events[e].append(mm)
    tab=[]
    for e in events:
        q=np.array(e,float) if len(e) else np.zeros((1,len(MOMENT_NAMES)))
        tab.extend(q.mean(0));tab.extend(q.max(0))
    tab.extend(len(e) for e in events)
    return np.asarray(values,np.float32),np.asarray(cat,np.int16),cards,np.asarray(tab,np.float32)
