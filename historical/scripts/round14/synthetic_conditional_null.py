"""Exhaustive mathematical checks, NOT synthetic-to-Kaggle performance claims."""
import itertools,json
from pathlib import Path
import numpy as np
from opportunity_likelihood import sparse_mixture,episodic_hmm

def run():
    rows=[];old=new=0.
    for first in (0,1):
        second_p=.8 if first else .1
        for second in (0,1):
            mass=(.2 if first else .8)*(second_p if second else 1-second_p)
            a=first*second-.2*second_p;b=first*(second-second_p);old+=mass*a;new+=mass*b
            rows.append({'first':first,'second':second,'mass':mass,'p_second':second_p,'old_product_residual':a,'predictable_gate_residual':b})
    p=np.array([.2,.4,.7]);mix=hmm=0.
    for ys in itertools.product([0,1],repeat=3):
        y=np.array(ys);mass=np.prod(np.where(y,p,1-p))
        mix+=mass*np.exp(sparse_mixture(y,p)['log_bf']);hmm+=mass*np.exp(episodic_hmm(y,p)['log_bf'])
    result={'state':'EXHAUSTIVE_SYNTHETIC_DIAGNOSTIC_NOT_KAGGLE_GAIN','normal_sequential_table':rows,'expected_old_residual':old,'expected_corrected_residual':new,'mixture_LR_null_mean':mix,'HMM_LR_null_mean':hmm,'caveat':'Unit null means require the specified correct policy; estimated real-data q0 need calibration.'}
    O=Path(__file__).resolve().parents[1]/'results';(O/'synthetic_null.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':run()
