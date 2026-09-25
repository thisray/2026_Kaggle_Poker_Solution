"""R5-H1: dev's labelled set contains ZERO other_coordination, yet eval clearly has
that family (~17% of positives by R5-f1). Two explanations:
  (a) the labelled 1,860 pairs are a ~76% sample of dev positives and the ~117 hidden
      positives are ordinary DT/SP/CI that simply were not sampled;
  (b) other_coordination IS the hidden set - the organisers labelled only the three
      families they could confirm.
(b) would be a big deal: the hidden pairs would be dev-side examples of the fourth
family, letting us check the A-type hypothesis on dev directly.
Test: profile the hidden pairs with the DT two-type event models (both orientations)
and with the 3-class family classifier, against labelled positives and confirmed
negatives."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, os, json
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18'); sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917')
import ci_censored_event as C
from pairfeat import build
from pairfeat2 import build2
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'
FAMS=['directed_transfer','soft_play','coordinated_isolation']
T=pd.read_parquet(f'{O}/m5_both_train_oof.parquet')
u=T[T.label==-1].pivot_table(index='key',columns='src',values='oof')
sub=[c for c in u.columns if c.startswith('devsub')]
hid=u[(u[sub].min(axis=1)>=0.3)].index
print('hidden-positive keys (oof>=0.3 in every devsub):',len(hid))
d=T[T.src=='devsub11'].copy().sort_values('key',kind='mergesort').reset_index(drop=True)
t=pd.read_parquet(f'{O}/ptab_devsub11.parquet'); t['key']=t.p_lo*12000+t.p_hi; t=t.set_index('key').loc[d.key].reset_index()
X=pd.concat([build(t),build2(t)],axis=1)
pos=(d.y.values==1); neg=(d.label.values==0); ishid=d.key.isin(set(hid)).values
print('in devsub11 table: labelled positives',int(pos.sum()),'confirmed negatives',int(neg.sum()),'hidden',int(ishid.sum()))
yf=pd.Series(d.fam.values).map({f:i for i,f in enumerate(FAMS)}).values
prob=np.zeros((len(d),3))
for f in range(5):
    tr=pos&(d.fold.values!=f); va=d.fold.values==f
    m=lgb.train(dict(objective='multiclass',num_class=3,learning_rate=.05,num_leaves=15,min_data_in_leaf=10,feature_fraction=.6,verbose=-1,num_threads=8),lgb.Dataset(X[tr],yf[tr].astype(int)),300)
    prob[va]=m.predict(X[va])
mx=prob.max(1); am=np.array(FAMS)[prob.argmax(1)]
print()
print('3-class family classifier, max posterior (a genuinely 4th family should look LESS like any of the three):')
for nm,msk in (('labelled positives',pos),('hidden positives',ishid),('confirmed negatives',neg)):
    print(f'  {nm:22s} n={int(msk.sum()):5d}  mean max-prob {mx[msk].mean():.3f}  quartiles {np.round(np.percentile(mx[msk],[25,50,75]),3)}  argmax mix {pd.Series(am[msk]).value_counts(normalize=True).round(3).to_dict()}')
print()
print('by true family (labelled positives):')
for f in FAMS:
    m2=pos&(d.fam.values==f); print(f'  {f:24s} n={int(m2.sum()):4d}  mean max-prob {mx[m2].mean():.3f}  argmax mix {pd.Series(am[m2]).value_counts(normalize=True).round(3).to_dict()}')
json.dump(dict(n_hidden=int(len(hid)),maxprob=dict(labelled=float(mx[pos].mean()),hidden=float(mx[ishid].mean()),negative=float(mx[neg].mean())),
   argmax_mix_hidden=pd.Series(am[ishid]).value_counts(normalize=True).round(4).to_dict(),
   argmax_mix_labelled=pd.Series(am[pos]).value_counts(normalize=True).round(4).to_dict()),open(f'{O}/r5/h1_hidden_family.json','w'),indent=1)
