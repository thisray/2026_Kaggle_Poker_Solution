#!/usr/bin/env python3
"""Extract R18 CI inputs from existing GB10 R1/R15 arrays (no new policy inference).
Gameplay definitions mirror R3 s66_outcome_conditions.py / t5_ci_trigger.py.
The adapter has been source-reviewed and syntax-tested here, but the raw GB10
arrays are not in the R3 ZIP. Run the documented one-time dev feature parity
check on GB10 before inference. No submission or remote side effects.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

@njit(cache=True)
def sequence(H,S,T,off,seat,street,Y,pa,dealt):
    out=np.zeros((len(H),11),np.int64)
    for row in range(len(H)):
        h=H[row];f=-1
        for k in range(off[h],off[h+1]):
            if street[k]!=0:break
            if seat[k]==S[row] or seat[k]==T[row]:f=k;break
        if f<0:out[row,0]=-1;continue
        me=seat[f];partner=T[row] if me==S[row] else S[row]
        out[row,0]=0 if me==S[row] else 1;out[row,1]=Y[f];out[row,10]=pa[f]
        alive=dealt[row].copy();nvol=0
        for k in range(off[h],f):
            if Y[k]==0:alive[seat[k]]=0
            elif seat[k]!=S[row] and seat[k]!=T[row] and Y[k]>=2:nvol+=1
        out[row,9]=nvol
        for s in range(6):
            if alive[s]==1 and s!=S[row] and s!=T[row]:out[row,4]+=1
        pf=-1;nf=0;nc=0;last_st=0
        for k in range(f+1,off[h+1]):
            last_st=street[k]
            if street[k]!=0:continue
            s=seat[k]
            if s==partner and pf<0:pf=Y[k]
            if s!=S[row] and s!=T[row]:
                if Y[k]==0:nf+=1;alive[s]=0
                elif Y[k]>=2:nc+=1
        out[row,2]=pf;out[row,3]=nf;out[row,5]=nc;out[row,6]=1 if last_st==0 else 0
        for s in range(6):
            if alive[s]==1 and s!=S[row] and s!=T[row]:out[row,7]+=1
    return out

def extract(out_dir: Path, keyed_hands: pd.DataFrame) -> pd.DataFrame:
    D=out_dir/'np';load=lambda n:np.load(D/f'{n}.npy',mmap_mode='r')
    sp=load('s_player');net=load('s_net');won=load('s_won');sd=load('s_sd');fold=load('s_folded');bb=load('h_bb')
    P=np.load(out_dir/'P_v1.npy',mmap_mode='r');R=np.load(out_dir/'R_v1.npy',mmap_mode='r')
    lines=(out_dir/'feature_names_v1.txt').read_text().splitlines();rn=lines[0][2:].split(',');pn=lines[1][2:].split(',')
    loc=pd.read_parquet(out_dir/'player_local_v1.parquet');mem=np.full((int(loc.pool.max())+1,30),-1,np.int64)
    mem[loc.pool.values,loc.local.values]=loc.player_gi.values
    M=keyed_hands.rename(columns={'sl':'slot'}).copy().reset_index(drop=True);H=M.h.to_numpy(dtype=np.int64);sl=M.slot.to_numpy(dtype=np.int64)
    a=mem[sl//900,(sl%900)//30];b=mem[sl//900,sl%30];spH=np.asarray(sp[H]);S=np.argmax(spH==a[:,None],axis=1);T=np.argmax(spH==b[:,None],axis=1);ix=np.arange(len(H))
    if not ((spH[ix,S]==a)&(spH[ix,T]==b)&(S!=T)).all():raise ValueError('Invalid player-slot / shared-hand mapping')
    bbs=np.asarray(bb[H],float);n=np.asarray(net[H],float);w=np.asarray(won[H]);d=np.asarray(sd[H]);f=np.asarray(fold[H])
    M['na']=n[ix,S]/bbs;M['nb']=n[ix,T]/bbs;M['pair_net']=M.na+M.nb
    M['wa']=w[ix,S]>0;M['wb']=w[ix,T]>0;M['pair_win']=M.wa|M.wb;M['sd_any']=d.sum(axis=1)>0
    M['fa']=f[ix,S]>0;M['fb']=f[ix,T]>0
    get=lambda i,j,c:np.asarray(R[H,i,j,rn.index(c)],float)
    M['iso']=get(S,T,'iso_ofold')+get(T,S,'iso_ofold');M['xfer']=(get(S,T,'flow')!=0)|(get(T,S,'flow')!=0)
    Y=np.load(out_dir/'dec_Y.npy',mmap_mode='r')
    O=sequence(H,S.astype(np.int64),T.astype(np.int64),load('a_off'),load('a_seat'),load('a_st'),Y,load('a_players_active'),(spH>=0).astype(np.int64))
    cols=['who','y1','y2','o_fold_after','o_in_at_trig','o_callraise_after','end_pre','o_alive_endpre','_unused','o_vol_before','pa_at_trig']
    for j,c in enumerate(cols):M[c]=O[:,j]
    eq=np.asarray(P[H,:,pn.index('pf_eq_rand')],np.float32);ls=np.asarray(P[H,:,pn.index('last_street')],np.float32)
    M['own']=np.where(M.who==0,eq[ix,S],eq[ix,T]);M['par']=np.where(M.who==0,eq[ix,T],eq[ix,S])
    ow=np.zeros(len(H),bool);os=np.zeros(len(H),bool)
    for j in range(6):
        outsiders=(j!=S)&(j!=T)&(spH[:,j]>=0);ow|=outsiders&(w[:,j]>0);os|=outsiders&(d[:,j]>0)
    M['out_won']=ow;M['out_sd']=os;M['all_out_folded']=(~ow)&(~os);M['both_flop']=np.minimum(ls[ix,S],ls[ix,T])>=1
    M['ts']=np.asarray(load('h_ts')[H]);return M.drop(columns='_unused')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/home/thisray/projects/260916_Kaggle_Poker_artifacts');p.add_argument('--base');p.add_argument('--scored');p.add_argument('--out-dir',required=True);p.add_argument('--parity-dev');a=p.parse_args()
    root=Path(a.root);od=root/'opus_r1_20260917';D=od/'np';dst=Path(a.out_dir);dst.mkdir(parents=True,exist_ok=True)
    if a.parity_dev:
        from ci_censored_event import prepare,FEATURES
        truth=pd.read_parquet(a.parity_dev).rename(columns={'sl':'slot'});truth=truth[truth.fam=='coordinated_isolation'].reset_index(drop=True)
        rebuilt=extract(od,truth[['slot','h','fam','ev']]);left=prepare(truth);right=prepare(rebuilt)
        diffs={c:float(np.max(np.abs(left[c].astype(float)-right[c].astype(float)))) for c in FEATURES+['ts']}
        (dst/'adapter_parity.json').write_text(json.dumps(diffs,indent=2));assert max(diffs.values())<1e-6,diffs
        print('R3 dev gameplay parity PASS');return
    if not a.base:p.error('--base is required for eval extraction')
    base=pd.read_csv(a.base,dtype={'pair_id':str});sc=pd.read_csv(a.scored or root/'round15_campaign/scored_tabicl_rank_blend.csv')
    required={'pair_id','slot','hand_id','score'}
    if not required<=set(sc):raise ValueError(f'R15 scored table needs {required}; found {list(sc)}')
    ids=set(base.loc[base.predicted_behavior=='coordinated_isolation','pair_id']);sc=sc[sc.pair_id.isin(ids)].copy()
    idx=pd.read_parquet(D/'hand_index.parquet');hmap=idx.set_index('hand_id').hi;sc['h']=sc.hand_id.map(hmap)
    if sc.h.isna().any():raise ValueError('Missing hand IDs')
    sc['h']=sc.h.astype('int64');sc=sc.sort_values(['slot','score'],ascending=[True,False],kind='stable').groupby('slot',sort=False).head(20).copy()
    sc['r']=sc.groupby('slot',sort=False).cumcount()+1
    if not sc.groupby('slot').size().ge(5).all():raise ValueError('Insufficient candidates')
    loc=pd.read_parquet(od/'player_local_v1.parquet');mem=np.full((int(loc.pool.max())+1,30),-1,np.int64);mem[loc.pool.values,loc.local.values]=loc.player_gi.values
    sp=np.load(D/'s_player.npy',mmap_mode='r');ht=np.load(D/'h_table.npy');ph=np.load(D/'h_phase.npy')
    keyed=[]
    for pool,pg in sc[['slot','pair_id']].drop_duplicates().assign(pool=lambda x:x.slot//900).groupby('pool'):
        hs=np.flatnonzero((ht==pool)&(ph==1));seats=np.asarray(sp[hs])
        for row in pg.itertuples(index=False):
            sl=int(row.slot);pa=mem[pool,(sl%900)//30];pb=mem[pool,sl%30]
            use=(seats==pa).any(axis=1)&(seats==pb).any(axis=1)
            keyed.append(pd.DataFrame({'slot':sl,'h':hs[use],'pair_id':row.pair_id,'fam':'coordinated_isolation'}))
    if not keyed:raise ValueError('No CI-routed eval pairs')
    full=extract(od,pd.concat(keyed,ignore_index=True));full.to_parquet(dst/'ci_eval_full.parquet',index=False)
    sc[['pair_id','slot','h','hand_id','r','score']].to_parquet(dst/'ci_eval_candidates.parquet',index=False)
    print(json.dumps({'CI_pairs':int(sc.slot.nunique()),'shared_hands':len(full),'candidates':len(sc),'raw_eval_executed':True,'not_submitted':True}))
if __name__=='__main__':main()
