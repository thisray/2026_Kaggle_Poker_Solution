"""Hierarchical own-card entry policy on observed decisions.
This is an incremental 169-class lookup view, not a substitute for the existing
policy model. Exclude the outer pools from FIT, pass only pre-decision context.
Inputs: hole1,hole2 integer rank*4+suit; position 0..5; faced_raise 0/1; entered 0/1.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd


def hole_class(c1,c2):
    a=np.asarray(c1,int);b=np.asarray(c2,int)
    if np.any((a<0)|(a>=52)|(b<0)|(b>=52)|(a==b)):raise ValueError('Invalid distinct hole cards')
    hi=np.maximum(a//4,b//4);lo=np.minimum(a//4,b//4);su=(a%4==b%4)
    # Conventional 13x13 class grid: diagonal pairs, opposite triangles suited/offsuit.
    return np.where(hi==lo,hi*13+lo,np.where(su,hi*13+lo,lo*13+hi)).astype(int)


class EntryPolicy:
    def __init__(self,smoothing=20.):
        self.smoothing=float(smoothing)
        if self.smoothing<=0:raise ValueError('Positive smoothing')
    @staticmethod
    def keys(d):
        cls=hole_class(d.hole1,d.hole2)
        pos=d.position.to_numpy(int);faced=d.faced_raise.to_numpy(int)
        if np.any((pos<0)|(pos>5)) or np.any((faced!=0)&(faced!=1)):raise ValueError('Context out of range')
        return cls,(cls*6+pos)*2+faced
    def fit(self,d,heldout_pools=()):
        if set(d.pool)&set(heldout_pools):raise ValueError('Held-out pools in background policy fit')
        cls,key=self.keys(d);y=d.entered.to_numpy(float)
        if not np.isin(y,[0,1]).all():raise ValueError('Entry label must be observed binary action')
        self.global_p=(y.sum()+1)/(len(y)+2)
        cn=np.bincount(cls,minlength=169);cs=np.bincount(cls,weights=y,minlength=169)
        self.class_p=(cs+self.smoothing*self.global_p)/(cn+self.smoothing)
        n=np.bincount(key,minlength=169*12);s=np.bincount(key,weights=y,minlength=169*12)
        self.prob=(s+self.smoothing*np.repeat(self.class_p,12))/(n+self.smoothing)
        self.n=n;return self
    def predict(self,d):
        _,key=self.keys(d);return np.clip(self.prob[key],1e-5,1-1e-5)
    def save(self,out):
        out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
        np.savez(out,prob=self.prob,n=self.n,class_p=self.class_p,global_p=self.global_p,smoothing=self.smoothing)
    @classmethod
    def load(cls,path):
        p=np.load(path);m=cls(float(p['smoothing']))
        for name in ['prob','n','class_p','global_p']:setattr(m,name,p[name])
        return m


def paired_entry_features(p_first,p_second,entered_first,entered_second):
    """p_second must condition on PUBLIC history before the second decision.
    The product is a conditional sequential diagnostic, not a proven null for
    shared-card dependence or a coordinated-pair probability.
    """
    p=np.clip(np.asarray([p_first,p_second],float),1e-5,1-1e-5)
    y=np.asarray([entered_first,entered_second],float)
    if not np.isin(y,[0,1]).all():raise ValueError('Observed entry bits required')
    loose=-np.log(p)*y;res=y-p
    return dict(entry_surprise_sum=float(loose.sum()),entry_surprise_min=float(loose.min()),
                both_entered=float(y.prod()),second_surprise=float(loose[1]),
                residual_product=float(res.prod()),expected_both_conditional=float(p.prod()))
