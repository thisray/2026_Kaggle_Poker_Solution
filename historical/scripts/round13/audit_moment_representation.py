"""Structural tests on the supplied R8 replay encoder. No raw-data score claim."""
import sys,json,hashlib
from pathlib import Path
import numpy as np

def run(legacy,out):
 sys.path.insert(0,str(Path(legacy).resolve()))
 import replay
 rng=np.random.default_rng(71);tests=[]
 for trial in range(40):
  cards=rng.choice(52,17,replace=False);holes=cards[:12].reshape(6,2);board=cards[12:]
  records=[]
  for t in range(12):
   i=int(rng.integers(6));st=t//3;act=int(rng.integers(6));la=int(rng.integers(-1,6));tc=float(rng.uniform(0,20))
   alive=np.ones(6,dtype=bool);can=alive.copy();amt=float(rng.uniform(0,30));aggr=act in [3,4] or act==5 and amt>tc
   g=np.vstack([replay.card_geometry(holes[s],board[:0 if st==0 else st+2]) for s in range(6)])
   records.append(dict(i=i,st=st,act=act,la=la,tc=tc,pot=50.,amt=amt,stack=100.,bb=2.,btn=0,
       alive=alive,can=can,raise_proxy=1.,raises=1,last_full=2.,total=np.full(6,10.),probs=np.array([.2,.3,.3,.2]),
       prob_known=True,eq96=.3,eqla96=.4,stacks=np.full(6,100.),bets=np.full(6,5.),faced=np.full(6,3,dtype=np.int64),
       full=np.full(6,.3),hu=np.full((6,6),.5),geom=g,aggr=aggr,y=0 if act==0 else 1 if act==1 else 3 if aggr else 2))
  # Arbitrary record-level inputs test encoder symmetry, not physical game validity.
  a=replay.view(records,holes,board,0,1)[3];b=replay.view(records,holes,board,1,0)[3]
  tests.append(float(np.max(np.abs(a-b))))
 assert max(tests)==0
 result={'scope':'Code-structural invariance test over 40 constructed record streams; NOT real gameplay or model validation.',
         'tab_columns_per_orientation':len(replay.TAB_NAMES),'r11_moment_columns':2*len(replay.TAB_NAMES),
         'orientation_tab_max_difference':max(tests),'conclusion':'The tab summary is A/B-symmetric. Concatenating orientation mean and max duplicates the 294 coordinates. This is not a score ceiling; retain oriented nonlinear event features before symmetrization.',
         'missing_axes_observed_in_source':['No street axis in TAB_NAMES','No actor role axis in TAB_NAMES','Only per-event coordinate means and maxima; no general multivariate event distribution'],
         'source_sha256':hashlib.sha256((Path(legacy)/'replay.py').read_bytes()).hexdigest()}
 Path(out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':run(sys.argv[1],sys.argv[2])
