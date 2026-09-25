"""Fixed candidate-comparison experiment on supplied OOF; no raw/nested claims.
Direct within-pair reference-vs-other comparisons; antisymmetric expected wins.
No network or submission. IDs and truth counts are never input features.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
from catboost import CatBoostClassifier,Pool
SCORES=['u_r5b','z_cat','z_lr','z_lr11','rs_cat','rs_ranker','rs_blend','re_blend']
ENTRY=['en_surprise_sum','en_surprise_min','en_both_entered','en_second_surprise','en_resid_product','en_expected_both','en_second_entered','en_first_entered']

def pair_metrics(d,s):
    rows=[];y=d.ev.to_numpy()
    for slot,ix in d.groupby('slot',sort=False).indices.items():
        order=ix[np.argsort(-s[ix],kind='stable')[:5]];t=y[order]
        rows.append(dict(slot=int(slot),pool=int(d.pool.iloc[ix[0]]),fold=int(d.fold.iloc[ix[0]]),E=float(np.sum(t*np.cumsum(t)/np.arange(1,len(t)+1))/min(int(d.m_p.iloc[ix[0]]),5))))
    return pd.DataFrame(rows)

def bootstrap(base,new,reps=3000):
    d=base.merge(new,on=['slot','pool','fold'],suffixes=('_base','_new'));d['delta']=d.E_new-d.E_base
    g=d.groupby('pool').delta.agg(['sum','count']);r=np.random.default_rng(918)
    ids=r.integers(0,len(g),(reps,len(g)));means=g['sum'].to_numpy()[ids].sum(1)/g['count'].to_numpy()[ids].sum(1)
    return dict(delta=float(d.delta.mean()),CI95=np.quantile(means,[.025,.975]).tolist(),wins=int((d.delta>1e-10).sum()),losses=int((d.delta<-1e-10).sum()),per_fold=d.groupby('fold').delta.mean().to_dict())

def run(input,out,threads=2,seed=20260918,only_entry=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(input); assert not d.duplicated(['slot','hand_id']).any()
    assert d.groupby('pool').fold.nunique().max()==1
    config=dict(protocol='frozen_upstream_exploration',seed=seed,iterations=180,depth=5,learning_rate=.04,l2_leaf_reg=20.,feature_sets=['score_comparison','score_entry_comparison'],decode='antisymmetric_expected_wins',no_outer_weight_search=True)
    (out/'preregistered.json').write_text(json.dumps(config,indent=2))
    groups=list(d.groupby('slot',sort=False).indices.values());y=d.ev.to_numpy();f=d.fold.to_numpy();base=d.rs_blend.to_numpy()
    bm=pair_metrics(d,base);bm.to_csv(out/'baseline_pairs.csv',index=False);res={}
    settings=[('score_entry_comparison',SCORES+ENTRY)] if only_entry else [('score_comparison',SCORES),('score_entry_comparison',SCORES+ENTRY)]
    for name,cols in settings:
        x=d[cols].to_numpy(float);x=np.nan_to_num(x,nan=0,posinf=30,neginf=-30).clip(-30,30)
        ranks=np.zeros_like(x);context=np.zeros((len(d),len(cols)*2))
        for ix in groups:
            ranks[ix]=np.column_stack([rankdata(x[ix,j],method='average')/len(ix) for j in range(x.shape[1])]);context[ix]=np.r_[x[ix].mean(0),x[ix].std(0)]
        x=np.c_[x,ranks];ai=[];bi=[];w=[]
        for ix in groups:
            p=ix[y[ix]==1];n=ix[y[ix]==0]
            if len(p)*len(n)==0:continue
            a=np.repeat(p,len(n));b=np.tile(n,len(p));ai.extend(a);bi.extend(b);w.extend(np.full(len(a),1/len(a)))
        ai=np.asarray(ai);bi=np.asarray(bi);w=np.asarray(w);aa=np.r_[ai,bi];bb=np.r_[bi,ai];target=np.r_[np.ones(len(ai)),np.zeros(len(ai))];weights=np.r_[w,w]/2
        def features(a,b):return np.c_[x[a]-x[b],np.abs(x[a]-x[b]),(x[a]+x[b])/2,context[a]].astype(np.float32)
        XX=features(aa,bb);pred=np.zeros(len(d));fold_log=[]
        for fold in sorted(np.unique(f)):
            tr=f[aa]!=fold
            m=CatBoostClassifier(iterations=180,depth=5,learning_rate=.04,l2_leaf_reg=20,random_seed=seed,thread_count=threads,verbose=False,allow_writing_files=False)
            m.fit(Pool(XX[tr],target[tr],weight=weights[tr]))
            va_a=[];va_b=[]
            for ix in groups:
                if f[ix[0]]!=fold:continue
                for a in ix:
                    for b in ix:
                        if a!=b:va_a.append(a);va_b.append(b)
            a=np.asarray(va_a);b=np.asarray(va_b)
            z=m.predict(features(a,b),prediction_type='RawFormulaVal');zr=m.predict(features(b,a),prediction_type='RawFormulaVal')
            pred+=np.bincount(a,weights=expit((z-zr)/2),minlength=len(d))
            log=dict(fold=int(fold),train_comparisons=int(tr.sum()),eval_comparisons=len(a));fold_log.append(log);print(name,log,flush=True)
        pm=pair_metrics(d,pred);pm.to_csv(out/f'{name}_pairs.csv',index=False)
        z=d[['slot','hand_id','pool','fold','ev','m_p']].copy();z['base']=base;z['candidate']=pred;z.to_csv(out/f'{name}_oof.csv.gz',index=False)
        res[name]=dict(E=float(pm.E.mean()),**bootstrap(bm,pm),folds=fold_log);(out/'results.json').write_text(json.dumps(res,indent=2));print(name,res[name],flush=True)
    return res
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=20260918);p.add_argument('--only-entry',action='store_true');run(**vars(p.parse_args()))
