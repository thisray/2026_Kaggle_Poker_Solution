"""Retain partner-vs-each-outsider information instead of only outsider mean.
This computes predictive contrasts, not causal identification or membership truth.
"""
import numpy as np

def profile(logp_self,logp_partner,logp_outsider,valid_outsider):
    a,b,o,m=map(np.asarray,(logp_self,logp_partner,logp_outsider,valid_outsider))
    if o.ndim!=2 or m.shape!=o.shape or len(a)!=len(o) or b.shape!=a.shape:raise ValueError('Shape mismatch')
    n=m.sum(1);masked=np.where(m,o,-np.inf)
    best=masked.max(1);mean=np.divide(np.where(m,o,0).sum(1),n,out=np.zeros(len(n),float),where=n>0)
    all_sources=np.c_[b,masked];maxv=all_sources.max(1,keepdims=True)
    prob=np.exp(all_sources-maxv);prob/=prob.sum(1,keepdims=True)
    ent=-(prob*np.log(np.clip(prob,1e-300,1))).sum(1)
    return {'partner_vs_self':b-a,'partner_vs_mean_outsider':np.where(n>0,b-mean,np.nan),
            'partner_vs_best_outsider':np.where(n>0,b-best,np.nan),
            'partner_source_rank':1+(m&(o>b[:,None])).sum(1),
            'source_entropy':ent,'outsiders_available':n}
