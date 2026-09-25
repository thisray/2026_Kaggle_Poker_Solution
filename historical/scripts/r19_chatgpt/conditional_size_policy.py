"""Pool-cross-fitted conditional betting-size likelihood. Implementation, not a executed Poker result.

NPZ input (export on GB10): X_self, X_partner [n,f] numeric PRE-ACTION
features with ONLY the own-card block replaced for X_partner; y [n] fixed
size-bin labels; pool [n]; is_reference [n] bool (normal/reference actions
only); optional decision_id. Rows must be VOLUNTARY bet/raise actions;
forced calls excluded. Exporter must distinguish all-in and min-raise masses.
No amount actually chosen, future board or future action may enter X.

Recommended bins: 0 all-in; 1 legal-minimum raise; remaining bins of
(amount-to_call)/(pot_before+to_call), with fixed cuts .33,.67,1,2,4.
Use rules to handle caps, not floating equality without chip precision.
Outputs conditional-on-aggressive-action size LR. Do NOT multiply another
aggressive-action likelihood into it twice.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--trees',type=int,default=250);args=p.parse_args();z=np.load(args.input,allow_pickle=False)
    xs,xp,y,g,ref=(z[k] for k in ['X_self','X_partner','y','pool','is_reference'])
    ref=ref.astype(bool);y=y.astype(int)
    if xs.shape!=xp.shape or len(xs)!=len(y) or len(g)!=len(y): raise ValueError('Shape mismatch')
    if xs.ndim!=2 or len(np.unique(g))<5: raise ValueError('Need numeric 2D features, >=5 pools')
    p0=np.zeros(len(y));p1=np.zeros(len(y));fold_id=np.zeros(len(y),int);meta=[]
    classes=np.unique(y[ref]);mapping={v:i for i,v in enumerate(classes)}
    if not set(np.unique(y)).issubset(mapping):raise ValueError('Reference data must cover all fixed size bins')
    yy=np.array([mapping[v] for v in y]);args.out.mkdir(parents=True,exist_ok=True)
    for f,(tr,va) in enumerate(GroupKFold(5,shuffle=True,random_state=260919).split(xs,groups=g)):
        tr=tr[ref[tr]]
        if len(tr)<100 or len(np.unique(yy[tr]))!=len(classes): raise ValueError(f'Insufficient reference/classes in fold {f}')
        model=lgb.LGBMClassifier(objective='multiclass',n_estimators=args.trees,
            learning_rate=.05,num_leaves=31,min_child_samples=100,reg_lambda=10,
            colsample_bytree=.9,n_jobs=args.threads,random_state=260919+f,verbosity=-1)
        model.fit(xs[tr],yy[tr]);a=model.predict_proba(xs[va]);b=model.predict_proba(xp[va])
        p0[va]=a[np.arange(len(va)),yy[va]];p1[va]=b[np.arange(len(va)),yy[va]];fold_id[va]=f
        mask=ref[va];meta.append({'fold':f,'train_reference':len(tr),'test_rows':len(va),
            'heldout_reference_logloss':float(-np.log(np.clip(p0[va][mask],1e-8,1)).mean()) if mask.any() else None})
        model.booster_.save_model(str(args.out/f'size_policy_fold{f}.txt'))
    lr=np.log(np.clip(p1,1e-8,1))-np.log(np.clip(p0,1e-8,1))
    extra={'decision_id':z['decision_id']} if 'decision_id' in z else {}
    np.savez_compressed(args.out/'size_likelihood_oof.npz',p_self=p0,p_partner=p1,log_ratio_size=lr,fold=fold_id,**extra)
    (args.out/'receipt.json').write_text(json.dumps({'folds':meta,'classes':classes.tolist(),
      'status':'size-model output; evidence and negative-control evaluation required'},indent=2))
if __name__=='__main__':main()
