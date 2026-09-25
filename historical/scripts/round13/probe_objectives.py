"""Fixed frozen-upstream probes; not fully nested end-to-end validation."""
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
import lightgbm as lgb,xgboost as xgb
from catboost import CatBoostClassifier,Pool
from sklearn.linear_model import LogisticRegression
from exact_map5 import objective_for
from oof_oracles import ap5
RAW=['u_r5b','z_cat','z_lr','rs_cat','rs_ranker']
def features(d,entry):
    parts=[d[RAW].to_numpy(float)]
    for c in RAW+['rs_blend']:
        a=d[c].to_numpy();r=np.zeros(len(d));z=np.zeros(len(d));gap=np.zeros(len(d))
        for ix in d.groupby('slot',sort=False).indices.values():
            r[ix]=rankdata(-a[ix],method='average');z[ix]=(a[ix]-a[ix].mean())/(a[ix].std()+1e-6);gap[ix]=a[ix]-np.max(a[ix])
        parts +=[r[:,None],z[:,None],np.clip(gap,-30,30)[:,None]]
    if entry:parts+=[d[[c for c in d if c.startswith('en_')]].to_numpy(float)]
    return np.nan_to_num(np.concatenate(parts,1),posinf=30,neginf=-30).astype('float32')
def pair_metrics(d,pred):
    vals=[]
    for slot,ix in d.groupby('slot',sort=False).indices.items():
        g=d.iloc[ix];y=g.ev.to_numpy()[np.argsort(-pred[ix],kind='stable')]
        vals.append((int(slot),int(g.pool.iloc[0]),int(g.fold.iloc[0]),ap5(y,int(g.m_p.iloc[0]))))
    return pd.DataFrame(vals,columns=['slot','pool','fold','E'])
def run(inp,out):
    d=pd.read_csv(inp).sort_values('slot',kind='stable').reset_index(drop=True);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cfg={'scope':'Frozen upstream, diagnostic only; upstream OOF features may carry indirect outer-label dependencies.',
         'methods':['xgb_map_residual','lgb_exact_map5_residual','cat_binary_residual'],
         'views':['scores','scores_entry'],'seed':20260918,'threads':2,'rounds':180,'no_hyperparameter_search':True}
    (out/'preregistered_config.json').write_text(json.dumps(cfg,indent=2))
    base=d.rs_blend.to_numpy(float);bp=pair_metrics(d,base);bp.to_csv(out/'baseline_pairs.csv',index=False)
    result=[];preds=d[['slot','pool','fold','hand_id','ev','m_p']].copy();preds['baseline']=base
    for entry in [False,True]:
      X=features(d,entry);y=d.ev.to_numpy(int);view='scores_entry' if entry else 'scores'
      for method in cfg['methods']:
        t0=time.monotonic();pred=np.full(len(d),np.nan)
        for f in sorted(d.fold.unique()):
          tr=d.fold.to_numpy()!=f;va=~tr;dt=d[tr];groups=dt.groupby('slot',sort=False).size().to_numpy();initial=base/3.
          if method=='xgb_map_residual':
            dm=xgb.DMatrix(X[tr],label=y[tr],base_margin=initial[tr]);dm.set_group(groups)
            m=xgb.train(dict(objective='rank:map',eval_metric='map@5',lambdarank_pair_method='topk',lambdarank_num_pair_per_sample=8,
                     max_depth=3,eta=.025,min_child_weight=10,reg_lambda=20,tree_method='hist',nthread=2,seed=20260918),dm,num_boost_round=180)
            pred[va]=m.predict(xgb.DMatrix(X[va],base_margin=initial[va]))
          elif method=='lgb_exact_map5_residual':
            truth=dt.groupby('slot',sort=False).m_p.first().to_numpy();ds=lgb.Dataset(X[tr],label=y[tr],group=groups,init_score=initial[tr])
            m=lgb.train(dict(objective=objective_for(groups,truth),learning_rate=.025,num_leaves=7,min_data_in_leaf=30,
                    lambda_l2=10.,num_threads=2,verbosity=-1,seed=20260918,deterministic=True,force_col_wise=True),ds,num_boost_round=180)
            pred[va]=initial[va]+m.predict(X[va])
          else:
            cal=LogisticRegression(C=10,max_iter=1000).fit(base[tr,None],y[tr]);ini=cal.decision_function(base[:,None])
            m=CatBoostClassifier(iterations=180,depth=3,learning_rate=.025,l2_leaf_reg=30,thread_count=2,random_seed=20260918,verbose=False,allow_writing_files=False)
            m.fit(Pool(X[tr],y[tr],baseline=ini[tr],weight=1/d.loc[tr,'m_p'].to_numpy()));pred[va]=ini[va]+.25*m.predict(X[va],prediction_type='RawFormulaVal')
        assert np.isfinite(pred).all();name=f'{method}_{view}';preds[name]=pred
        p=pair_metrics(d,pred);p['base']=bp.E;p['delta']=p.E-p.base;p.to_csv(out/f'{name}_pairs.csv',index=False)
        agg=p.groupby('pool').agg(n=('E','size'),delta=('delta','sum'));rng=np.random.default_rng(20260918)
        ix=rng.integers(0,len(agg),(4000,len(agg)));boot=agg.delta.to_numpy()[ix].sum(1)/agg.n.to_numpy()[ix].sum(1)
        row=dict(name=name,E=float(p.E.mean()),delta=float(p.delta.mean()),fold_delta=p.groupby('fold').delta.mean().to_dict(),
                 pool_bootstrap_delta95=np.quantile(boot,[.025,.975]).tolist(),wins=int((p.delta>1e-12).sum()),losses=int((p.delta< -1e-12).sum()),seconds=time.monotonic()-t0)
        result.append(row);(out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(row),flush=True)
    preds.to_csv(out/'predictions.csv.gz',index=False)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--input',required=True);a.add_argument('--out',required=True);x=a.parse_args();run(x.input,x.out)
