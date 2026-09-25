"""Build raw candidate tensors directly from existing opus arrays.
Supports the supplied dev CSV; evaluation candidates need the same key columns
(or normalize them with export_candidates.py). No raw data are uploaded anywhere.
"""
from __future__ import annotations
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from replay import Arrays,replay_hand,view,NUMERIC_NAMES,TAB_NAMES

def read_table(path):
    p=Path(path)
    if p.exists():return pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p)
    if p.with_suffix('.csv').exists():return pd.read_csv(p.with_suffix('.csv'))
    raise FileNotFoundError(path)

def prepare(candidate_file,np_dir,out,exact=False,pair_offset=0,pair_limit=0):
    start=time.monotonic();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'meta.csv').exists():raise FileExistsError(f'{out} already contains a pack; choose a new output')
    d=read_table(candidate_file)
    required={'slot','pool','pair_player_lo','pair_player_hi','hand_id','u_r5b'}
    if required-set(d):raise ValueError(f'Missing {sorted(required-set(d))}; use export_candidates.py')
    if d.duplicated(['slot','hand_id']).any():raise ValueError('Duplicate candidate key')
    ps=np.sort(d.slot.unique());ps=ps[pair_offset:(pair_offset+pair_limit) if pair_limit else None]
    d=d[d.slot.isin(ps)].sort_values(['slot','hand_id'],kind='stable').reset_index(drop=True)
    if len(d)==0:raise ValueError('No candidates selected')
    if 'fold' in d and d.groupby('pool').fold.nunique().max()>1:raise ValueError('Pool fold split')
    a=Arrays(np_dir);hi=read_table(Path(np_dir)/'hand_index.parquet');pi=read_table(Path(np_dir)/'player_index.parquet')
    hm=hi.set_index('hand_id').hi;pm=pi.set_index('player_id').pi
    hh=d.hand_id.map(hm);lo=d.pair_player_lo.map(pm);high=d.pair_player_hi.map(pm)
    if hh.isna().any() or lo.isna().any() or high.isna().any():raise ValueError('Candidate IDs absent from existing indices')
    h=hh.to_numpy(int);pl=lo.to_numpy(int);ph=high.to_numpy(int)
    if 'ev' in d and not (a['h_phase'][h]==0).all():raise ValueError('Training labels attached to non-development hands')
    lengths=np.asarray(a['a_off'][h+1]-a['a_off'][h],int);L=int(lengths.max());N=len(d)
    if L<=0 or L>1024:raise ValueError(f'Unexpected action length {L}; inspect before allocating')
    arrays={}
    shapes={'num':(N,2,L,len(NUMERIC_NAMES)),'cat':(N,2,L,4),'cards':(N,2,4,17),
            'valid':(N,L),'tab':(N,2,len(TAB_NAMES))}
    types={'num':'float32','cat':'int16','cards':'int16','valid':'bool','tab':'float32'}
    for name,shape in shapes.items():
        arrays[name]=np.lib.format.open_memmap(out/f'{name}.npy',mode='w+',dtype=types[name],shape=shape)
        arrays[name][:]=(-1 if name=='cards' else 0)
    issue={'max_stack_error':0.,'illegal_actors':0};unique=0
    groups={}
    for r,hh in enumerate(h):groups.setdefault(int(hh),[]).append(r)
    for hh,rows in groups.items():
        rec,holes,board,stats=replay_hand(hh,a,exact=exact);unique+=1
        issue['max_stack_error']=max(issue['max_stack_error'],stats['max_stack_error']);issue['illegal_actors']+=stats['illegal_actors']
        for r in rows:
            seats=np.asarray(a['s_player'][hh]);ia=np.flatnonzero(seats==pl[r]);ib=np.flatnonzero(seats==ph[r])
            if len(ia)!=1 or len(ib)!=1 or ia[0]==ib[0]:raise ValueError(f'Non-shared hand at row {r}')
            for v,(A,B) in enumerate([(int(ia[0]),int(ib[0])),(int(ib[0]),int(ia[0]))]):
                num,cat,cards,tab=view(rec,holes,board,A,B)
                arrays['num'][r,v,:len(rec)]=num;arrays['cat'][r,v,:len(rec)]=cat
                arrays['cards'][r,v]=cards;arrays['tab'][r,v]=tab
            arrays['valid'][r,:len(rec)]=True
        if unique%250==0:print(f'prepared unique hands {unique}/{len(groups)}',flush=True)
    for x in arrays.values():x.flush()
    d.to_csv(out/'meta.csv',index=False)
    receipt={'rows':N,'pairs':len(ps),'unique_hands':unique,'max_actions':L,
             'fraction_actions_over_32':float((lengths>32).mean()),'numeric_dim':len(NUMERIC_NAMES),
             'tab_dim_per_view':len(TAB_NAMES),'exact_postflop':exact,'seconds':time.monotonic()-start,
             'replay_checks':issue,'numeric_names':NUMERIC_NAMES,'tab_names':TAB_NAMES,
             'input_sha256':hashlib.sha256(Path(candidate_file).read_bytes()).hexdigest(),
             'upstream_validation':'frozen_upstream_unless_caller_rebuilt_it',
             'semantics':'Retrospective evidence encoder; raise opportunity is a proxy, not a certified full legal engine.'}
    (out/'pack.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if not k.endswith('names')},indent=2))
    return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidates',required=True);p.add_argument('--np-dir',required=True);p.add_argument('--out',required=True)
    p.add_argument('--exact',action='store_true');p.add_argument('--pair-offset',type=int,default=0);p.add_argument('--pair-limit',type=int,default=0)
    x=p.parse_args();prepare(x.candidates,x.np_dir,x.out,x.exact,x.pair_offset,x.pair_limit)
if __name__=='__main__':main()
