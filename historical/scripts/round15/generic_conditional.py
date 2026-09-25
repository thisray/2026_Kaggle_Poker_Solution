"""Cross-fitted extra-information probe; outputs features, not collusion truth.
Input action CSV: pair_id, hand_id, action_no, actor, pool, fold, action_class,
own_* numeric inputs, partner_* numeric inputs. Optional matched_normal boolean.
Current action, amount, hidden partner cards MUST NOT appear in own_* columns.
Use development only for supervised/model-selection experiments.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer


def probability(model,x):
    ans=np.full((len(x),4),1e-6);ans[:,model.classes_.astype(int)]=model.predict_proba(x)
    return ans/ans.sum(1,keepdims=True)


def run(path,out,iterations=100):
    d=pd.read_csv(path);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    needed={'pair_id','hand_id','action_no','actor','pool','fold','action_class'}
    if not needed.issubset(d):raise ValueError(f'Missing {needed-set(d)}')
    if d.groupby('pool').fold.nunique().max()!=1:raise ValueError('Pool crosses folds')
    own=[c for c in d if c.startswith('own_')];partner=[c for c in d if c.startswith('partner_')]
    if not own or not partner:raise ValueError('Explicit information views required')
    y=d.action_class.to_numpy(int)
    if not np.isin(y,[0,1,2,3]).all():raise ValueError('Action encoding')
    pred0=np.zeros((len(d),4));pred1=pred0.copy();source=[]
    for f in sorted(d.fold.unique()):
        tr=d.fold.ne(f).to_numpy();va=~tr
        # Candidate expansion repeats the same action for different pair roles.
        # The null model counts each actor decision only once.
        dedup=d[tr].drop_duplicates(['hand_id','action_no','actor']).index
        imp0=SimpleImputer(strategy='median',keep_empty_features=True).fit(d.loc[dedup,own])
        imp1=SimpleImputer(strategy='median',keep_empty_features=True).fit(d.loc[tr,own+partner])
        params={'max_iter':iterations,'learning_rate':.06,'max_leaf_nodes':15,'l2_regularization':10.,
                'min_samples_leaf':40,'early_stopping':False,'random_state':20260918}
        m0=HistGradientBoostingClassifier(**params).fit(imp0.transform(d.loc[dedup,own]),y[dedup])
        # In q1 each valid partner-conditional observation has a meaning; weighting
        # prevents decisions represented by many candidate pairs dominating loss.
        cnt=d[tr].groupby(['hand_id','action_no','actor']).pair_id.transform('size').to_numpy()
        m1=HistGradientBoostingClassifier(**params).fit(imp1.transform(d.loc[tr,own+partner]),y[tr],sample_weight=1/cnt)
        pred0[va]=probability(m0,imp0.transform(d.loc[va,own]));pred1[va]=probability(m1,imp1.transform(d.loc[va,own+partner]))
        source.append({'fold':int(f),'q0_unique_actions':len(dedup),'q1_rows':int(tr.sum())})
    idx=np.arange(len(d));d['q0_y']=pred0[idx,y];d['q1_y']=pred1[idx,y]
    d['ll_gain']=np.log(np.clip(d.q1_y,1e-6,1))-np.log(np.clip(d.q0_y,1e-6,1))
    d['q0_surprise']=-np.log(np.clip(d.q0_y,1e-6,1))
    # A descriptive held-out statistic; dependence and nuisance estimation make
    # it NOT a valid p-value or automatic collusion label.
    rows=[]
    for key,g in d.groupby('pair_id',sort=False):
        z=g.ll_gain.to_numpy();rows.append({'pair_id':key,'pool':g.pool.iloc[0],
            'fold':g.fold.iloc[0],'n':len(g),'gain_sum':z.sum(),'gain_mean':z.mean(),
            'gain_top3':np.sort(z)[-3:].mean(),'gain_sd':z.std(),
            'gain_standardized':z.mean()*np.sqrt(len(z))/(z.std()+1e-3)})
    pd.DataFrame(rows).to_csv(out/'pair_features.csv.gz',index=False)
    d.to_csv(out/'action_oof.csv.gz',index=False)
    report={'own_columns':own,'partner_columns':partner,'mean_log_gain':float(d.ll_gain.mean()),
        'folds':source,'interpretation':'feature probe; additional predictability is not proof of collusion',
        'needed_next':'matched/deck-consistent negative controls and leave-one-family-out evaluation'}
    if 'matched_normal' in d:
        flag=d.matched_normal.astype(str).str.lower().map({'true':True,'false':False,'1':True,'0':False})
        if flag.isna().any():raise ValueError('matched_normal must be an explicit boolean')
        report['matched_normal_mean_gain']=float(d.loc[flag.astype(bool),'ll_gain'].mean())
    (out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    p.add_argument('--iterations',type=int,default=100);a=p.parse_args();run(a.input,a.out,a.iterations)
