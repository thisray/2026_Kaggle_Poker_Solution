"""R5 d1: exact dev composition of the E budget, and the AP@5 denominator structure."""
import pandas as pd, numpy as np, collections
RAW='/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw'
lab=pd.read_csv(f'{RAW}/development_labels.csv')
ev=pd.read_csv(f'{RAW}/development_evidence.csv')
print('dev labels', lab.shape, 'positives', int((lab.label==1).sum()))
print(lab[lab.label==1].behavior_family.value_counts())
print('label_status', lab.label_status.value_counts().to_dict())
g=ev.groupby('pair_id').size()
print('evidence pairs', len(g), 'rows', len(ev))
print('hands per pair distribution:', collections.Counter(g.values))
print('evidence family counts (pair level):', ev.groupby('pair_id').behavior_family.first().value_counts().to_dict())
# positives without evidence rows?
pos=set(lab[lab.label==1].pair_id); evp=set(ev.pair_id)
print('positives without evidence:', len(pos-evp), 'evidence pairs not positive:', len(evp-pos))
# per family: mean denominator min(5,n)
ev2=ev.groupby('pair_id').agg(n=('hand_id','size'), fam=('behavior_family','first'))
print(ev2.groupby('fam').n.agg(['size','mean','min','max']))
print('denominator min(5,n) mean by fam:')
ev2['den']=np.minimum(5,ev2.n)
print(ev2.groupby('fam').den.mean())
