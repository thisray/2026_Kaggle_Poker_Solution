"""Recompute P proxies with an exact pairwise pool-cluster bootstrap of rank AP.
This is an artifact diagnostic, not retraining and not hidden-label evaluation.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def ordered_ap(y, scores, keys):
    order = np.lexsort((np.asarray(keys), -np.asarray(scores)))
    yy = np.asarray(y)[order]
    return float(np.sum(yy * np.cumsum(yy) / np.arange(1, len(yy)+1)) / max(yy.sum(), 1))


def cluster_ap_draws(y, scores, keys, pool, counts, pool_levels):
    """Exact AP of duplicated pool samples. Duplicates share score and label."""
    idx = np.lexsort((np.asarray(keys), -np.asarray(scores)))
    y = np.asarray(y, int)[idx]
    inv = {p:i for i,p in enumerate(pool_levels)}
    g = np.array([inv[p] for p in np.asarray(pool)[idx]], int)
    pos = np.flatnonzero(y)
    prefix = np.zeros((len(pos), len(pool_levels)), int)
    hitprefix = np.zeros_like(prefix)
    c = np.zeros(len(pool_levels), int); h=c.copy(); j=0
    for i in range(len(y)):
        if y[i]:
            prefix[j] = c; hitprefix[j] = h; j+=1; h[g[i]]+=1
        c[g[i]]+=1
    ranks_before = prefix @ counts.T
    hits_before = hitprefix @ counts.T
    multiplicity = counts[:, g[pos]].T
    numerator = np.zeros(counts.shape[0], float)
    for rep in range(1, int(multiplicity.max())+1):
        numerator += np.where(multiplicity>=rep,
            (hits_before+rep)/(ranks_before+rep), 0.).sum(0)
    return numerator / np.maximum(multiplicity.sum(0),1)


def run(path, out, draws=2000):
    try: d=pd.read_parquet(path)
    except ImportError:
        from read_flat_parquet import read_flat
        d=read_flat(path)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    reports={};folds=[];rank_changes=[]
    for src in ['devsub11','devsub12']:
        t=d[d.src.eq(src)].reset_index(drop=True)
        assert not t.key.duplicated().any()
        if t.groupby('pool').fold.nunique().max()!=1:raise ValueError('Fold mismatch')
        levels=np.sort(t.pool.unique())
        rng=np.random.default_rng(20260918)
        counts=rng.multinomial(len(levels), np.full(len(levels),1/len(levels)),size=draws)
        y=t.y.to_numpy(int); scores=[t[c].to_numpy() for c in ['base_score','new_score']]
        point=[ordered_ap(y,s,t.key) for s in scores]
        bs=[cluster_ap_draws(y,s,t.key,t.pool,counts,levels) for s in scores]
        delta=bs[1]-bs[0]
        report={'rows':len(t),'known_positive':int(y.sum()),'known_positive_rate':float(y.mean()),
            'rank_AP_base':point[0],'rank_AP_new':point[1],'delta':point[1]-point[0],
            'pool_bootstrap_delta_interval_95':np.quantile(delta,[.025,.975]).tolist(),
            'bootstrap_fraction_positive':float(np.mean(delta>0)),
            'sklearn_AP':[float(average_precision_score(y,s)) for s in scores],
            'topK':{},'protocol':'known positives vs U-as-negative; current upstream pipeline, not clean validation'}
        for k in [50,100,200,450,1000,4000]:
            row={}
            for name,s in zip(['base','new'],scores):
                ii=np.lexsort((t.key,-s))[:k]
                row[name]={'known_positive':int(y[ii].sum()),'model_marked_hidden':int(t.hid.iloc[ii].sum()),
                    'known_negative':int(t.label.iloc[ii].eq(0).sum())}
            report['topK'][str(k)]=row
        for f,tf in t.groupby('fold'):
            ap=[ordered_ap(tf.y,tf[c],tf.key) for c in ['base_score','new_score']]
            folds.append({'src':src,'fold':int(f),'base':ap[0],'new':ap[1],'delta':ap[1]-ap[0]})
        for scope,mask in [('confirmed_PN',t.label.ge(0)),('drop_model_hidden',~t.hid)]:
            tmp=t[mask];report[scope]={'rows':len(tmp),'base':ordered_ap(tmp.y,tmp.base_score,tmp.key),
                    'new':ordered_ap(tmp.y,tmp.new_score,tmp.key)}
        ranks=[]
        for s in scores:
            rr=np.empty(len(t),int);rr[np.lexsort((t.key,-s))]=np.arange(1,len(t)+1);ranks.append(rr)
        for i in np.flatnonzero(y):
            rank_changes.append({'src':src,'key':int(t.key.iloc[i]),'pool':int(t.pool.iloc[i]),
                 'fold':int(t.fold.iloc[i]),'base_rank':int(ranks[0][i]),'new_rank':int(ranks[1][i]),
                 'rank_delta':int(ranks[1][i]-ranks[0][i]),'n':float(t.n.iloc[i])})
        reports[src]=report
    el=d[d.src.eq('eval_lab')]
    reports['eval_lab_training_assumption']={'rows':len(el),'all_y_zero':bool(el.y.eq(0).all()),
            'development_positive_rows_forced_zero':int(el.label.eq(1).sum()),
            'note':'Development label is not period-specific evaluation truth. No hidden-label inference here.'}
    reports['mean_delta']=float(np.mean([reports[s]['delta'] for s in ['devsub11','devsub12']]))
    (out/'p_recomputed.json').write_text(json.dumps(reports,indent=2))
    pd.DataFrame(folds).to_csv(out/'p_by_fold.csv',index=False)
    pd.DataFrame(rank_changes).to_csv(out/'known_positive_rank_changes.csv',index=False)
    print(json.dumps(reports,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    p.add_argument('--draws',type=int,default=2000);a=p.parse_args();run(a.input,a.out,a.draws)
