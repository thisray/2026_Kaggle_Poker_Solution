"""Fixed compact TabICLv2: CV, final fit, and inference. Run on GB10.
Public checkpoint auto-download only with explicit --download-public-checkpoint.
No package installation or Kaggle submission is performed by this program.
"""
from __future__ import annotations
import argparse,json,hashlib,gc
from pathlib import Path
import numpy as np
import pandas as pd


def read(path):
    return pd.read_parquet(path) if str(path).endswith('.parquet') else pd.read_csv(path)


def main(a):
    from tabicl import TabICLClassifier
    import tabicl,torch
    d=read(a.input);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    features=json.loads(Path(a.features).read_text())
    blocked={'pair_id','slot','hand_id','player_id','pool','fold','y','label','ev','m_p','phase','is_labeled'}
    if not isinstance(features,list) or not 1<=len(features)<=64 or set(features)&blocked:
        raise ValueError('Provide an explicit compact list of gameplay/score features, no identifiers/labels')
    X=d[features].replace([np.inf,-np.inf],np.nan)
    if a.mode=='predict':
        if not a.fitted:raise ValueError('Provide the self-produced fitted model')
        model=TabICLClassifier.load(a.fitted);score=model.predict_proba(X)[:,list(model.classes_).index(1)]
    else:
        if not {a.target,'pool','fold'}.issubset(d):raise ValueError('Target and pool folds required')
        if not d[a.target].isin([0,1]).all() or d.groupby('pool').fold.nunique().max()!=1:raise ValueError('Labels/folds')
        if not a.download_public_checkpoint and (not a.checkpoint or not Path(a.checkpoint).exists()):
            raise FileNotFoundError('Download the public checkpoint on GB10 or pass --download-public-checkpoint')
        y=d[a.target].to_numpy(int);score=np.zeros(len(d))
        folds=sorted(d.fold.unique()) if a.mode=='cv' else [None]
        for f in folds:
            tr=np.ones(len(d),bool) if f is None else d.fold.ne(f).to_numpy()
            va=np.ones(len(d),bool) if f is None else ~tr
            model=TabICLClassifier(n_estimators=a.ensemble,batch_size=min(a.ensemble,4),
                model_path=a.checkpoint,allow_auto_download=a.download_public_checkpoint,
                checkpoint_version='tabicl-classifier-v2-20260212.ckpt',device=a.device,
                use_fa3=False,random_state=a.seed,n_jobs=a.threads)
            model.fit(X[tr],y[tr]);score[va]=model.predict_proba(X[va])[:,list(model.classes_).index(1)]
            if f is None:model.save(str(out/'classifier.pkl'),save_model_weights=False,save_training_data=True)
            del model;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
            print('completed fold',f,flush=True)
    keep=[c for c in ['pair_id','slot','hand_id','pool','fold','ev','m_p',a.target] if c in d]
    keep=list(dict.fromkeys(keep));d[keep].assign(score=score).to_csv(out/'predictions.csv.gz',index=False)
    receipt={'mode':a.mode,'features':features,'checkpoint_version':'tabicl-classifier-v2-20260212.ckpt',
       'checkpoint_path':a.checkpoint,'tabicl_version':getattr(tabicl,'__version__','unknown'),
       'no_submission':True,'evidence_state':'EXECUTED_ON_INPUT_TABLE',
       'selection_scope':'predeclared features and existing folds; upstream scores need nested confirmation'}
    if a.checkpoint and Path(a.checkpoint).exists():receipt['checkpoint_sha256']=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cv','fit','predict'])
    for k in ['input','features','out']:p.add_argument('--'+k,required=True)
    p.add_argument('--target',default='ev');p.add_argument('--checkpoint');p.add_argument('--fitted')
    p.add_argument('--download-public-checkpoint',action='store_true');p.add_argument('--device',default='cuda')
    p.add_argument('--ensemble',type=int,default=8);p.add_argument('--threads',type=int,default=8)
    p.add_argument('--seed',type=int,default=20260918);main(p.parse_args())
