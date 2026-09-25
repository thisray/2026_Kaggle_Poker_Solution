"""R5-d6: on eval the evidence decoder is chosen by PREDICTED family, not the true
one, so E_known(eval) must be discounted by the routing error. Measure the family
classifier's OOF confusion on dev positives."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, json
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917')
from pairfeat import build
from pairfeat2 import build2
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'
FAMS=['directed_transfer','soft_play','coordinated_isolation']
T=pd.read_parquet(f'{O}/m5_both_train_oof.parquet'); out={}
for SUB in ('devsub11','devsub12'):
    d=T[T.src==SUB].copy()
    hid=set(T[(T.src==SUB)&(T.label==-1)&(T.oof>0.3)].key); d=d[~d.key.isin(hid)].sort_values('key',kind='mergesort').reset_index(drop=True)
    t=pd.read_parquet(f'{O}/ptab_{SUB}.parquet'); t['key']=t.p_lo*12000+t.p_hi; t=t.set_index('key').loc[d.key].reset_index()
    X=pd.concat([build(t),build2(t)],axis=1); pos=d.y.values==1; fam=d.fam.values
    yf=pd.Series(fam).map({f:i for i,f in enumerate(FAMS)}).values; prob=np.zeros((len(d),3))
    for f in range(5):
        tr=pos&(d.fold.values!=f); va=d.fold.values==f
        m=lgb.train(dict(objective='multiclass',num_class=3,learning_rate=.05,num_leaves=15,min_data_in_leaf=10,feature_fraction=.6,verbose=-1,num_threads=4),lgb.Dataset(X[tr],yf[tr].astype(int)),300)
        prob[va]=m.predict(X[va])
    pred=np.array(FAMS)[prob.argmax(1)]
    cm=pd.crosstab(pd.Series(fam[pos],name='true'),pd.Series(pred[pos],name='pred'))
    acc=float((pred[pos]==fam[pos]).mean())
    print(f'== {SUB}: family OOF accuracy on {int(pos.sum())} positives = {acc:.4f}'); print(cm.to_string())
    out[SUB]=dict(acc=acc,cm=cm.to_dict())
json.dump(out,open(f'{O}/r5/d6_routing.json','w'),indent=1)
