"""Compare integrated first-five posterior to enumeration of all seven latent events."""
from pathlib import Path
import itertools,json
import numpy as np,pandas as pd
from scipy.special import betaln,logsumexp,logit
from f4_random_effects import posterior,score
rng=np.random.default_rng(190919);lf=rng.normal(size=7);mu=.72;kappa=2.4;a=mu*kappa;b=(1-mu)*kappa
D=pd.DataFrame({'slot':1,'h':np.arange(7),'pool':0,'lf':lf,'l0':-4.,'ts':np.arange(7)})
z=np.array(list(itertools.product([0,1],repeat=7)));count=z.sum(1)
lw=betaln(a+count,b+7-count)-betaln(a,b)+z@lf;w=np.exp(lw-logsumexp(lw))
first=z*(np.cumsum(z,axis=1)<=5);truth=first.T@w
out,_=posterior(D,[logit(mu),np.log(kappa)],160)
err=float(np.max(abs(out.first5_plant_re-truth)));lrerr=float(abs(score([logit(mu),np.log(kappa)],D)-logsumexp(lw)))
assert err<1e-10 and lrerr<1e-10
result={'latent_assignments':128,'first5_posterior_max_abs_error':err,'marginal_log_likelihood_abs_error':lrerr,'status':'mathematical enumeration, not evidence validation'}
B=Path(__file__).resolve().parents[1];(B/'results/random_effects_math_test.json').write_text(json.dumps(result,indent=2));print(result)
