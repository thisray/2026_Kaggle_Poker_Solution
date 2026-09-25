"""Reproduce four pool splits and five fixed hand-gate models from compact cache."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from f4_hand_gate import matrix,fit,llr,MODELS
B=Path(__file__).resolve().parents[1];d=pd.read_csv(B/'results/f4_gate_input.csv.gz');lf=d.lf.to_numpy();rows=[]
for seed in [260919,11,29,47]:
 for model in MODELS:
  x=matrix(d,model);total=0.;folds=[]
  for tr,va in GroupKFold(5,shuffle=True,random_state=seed).split(d,groups=d.pool):
   beta=fit(x[tr],lf[tr]);v=float(llr(beta,x[va],lf[va]).sum());total+=v;folds.append(v)
  rows.append({'seed':seed,'model':model,'heldout_LLR':total,'fold_LLR':folds});print(rows[-1],flush=True)
(B/'results/hand_gate_reproduction.json').write_text(json.dumps(rows,indent=2))
