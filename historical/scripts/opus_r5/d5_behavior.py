"""R5-d5: predicted_behavior is used by NOTHING except B (P uses risk only, E uses
the evidence columns only), so it is an isolated lever. B_f = AP over
risk * 1[pred == f]: a wrong family on a HIGH-risk pair pushes every true f below it
down one rank, a wrong family on a low-risk pair costs nothing. How much B is left?"""
import numpy as np, pandas as pd, lightgbm as lgb, sys
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker/src')
from pokerlab.metrics import stable_ap
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917')
from pairfeat import build
from pairfeat2 import build2
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'
FAMS=['directed_transfer','soft_play','coordinated_isolation']
T=pd.read_parquet(f'{O}/m5_both_train_oof.parquet'); ref=T
out={}
for SUB in ('devsub11','devsub12'):
    d=T[T.src==SUB].copy()
    hid=set(ref[(ref.src==SUB)&(ref.label==-1)&(ref.oof>0.3)].key)
    d=d[~d.key.isin(hid)].sort_values('key',kind='mergesort').reset_index(drop=True)
    t=pd.read_parquet(f'{O}/ptab_{SUB}.parquet'); t['key']=t.p_lo*12000+t.p_hi; t=t.set_index('key').loc[d.key].reset_index()
    X=pd.concat([build(t),build2(t)],axis=1)
    pos=d.y.values==1; fam=d.fam.values; risk=d.oof.values; y=d.y.values
    yf=pd.Series(fam).map({f:i for i,f in enumerate(FAMS)}).values
    prob=np.zeros((len(d),3))
    for f in range(5):
        tr=pos&(d.fold.values!=f); va=d.fold.values==f
        m=lgb.train(dict(objective='multiclass',num_class=3,learning_rate=.05,num_leaves=15,min_data_in_leaf=10,feature_fraction=.6,verbose=-1,num_threads=4),lgb.Dataset(X[tr],yf[tr].astype(int)),300)
        prob[va]=m.predict(X[va])
    P=stable_ap(y,risk)
    def B(pred):
        aps={f:stable_ap((fam==f).astype(int),risk*(pred==f)) for f in FAMS}
        return float(np.mean(list(aps.values()))),{k:round(v,4) for k,v in aps.items()}
    r={}
    argmax=np.array(FAMS)[prob.argmax(1)]
    r['argmax']=B(argmax)
    # oracle family on the true positives only
    orc=argmax.copy(); orc[pos]=fam[pos]; r['oracle_on_positives']=B(orc)
    # 'none' for everything outside the top-K by risk (should be neutral)
    ordr=np.argsort(-risk)
    for K in (600,2000):
        p2=np.array(['none']*len(d),dtype=object); p2[ordr[:K]]=argmax[ordr[:K]]; r[f'top{K}_only']=B(p2)
    # abstain when the classifier is unsure and the pair is high risk (an uncertain high-risk
    # pair pollutes whichever block it joins; 'none' removes it from all three)
    for th in (0.45,0.55,0.65,0.8):
        p3=argmax.copy(); unsure=(prob.max(1)<th); p3[unsure]='none'; r[f'abstain{th}']=B(p3)
    # abstain only inside the top 1000 by risk
    for th in (0.5,0.7):
        p4=argmax.copy(); mask=np.zeros(len(d),bool); mask[ordr[:1000]]=True
        p4[mask&(prob.max(1)<th)]='none'; r[f'abstain_top1k_{th}']=B(p4)
    print(f'== {SUB}  P={P:.4f}  positives={int(y.sum())}')
    for k,(b,aps) in r.items(): print(f'   {k:22s} B={b:.4f}  dS={0.1*(b-r["argmax"][0]):+.5f}  {aps}')
    out[SUB]={k:v[0] for k,v in r.items()}
import json; json.dump(out,open(f'{O}/r5/d5_behavior.json','w'),indent=1)
