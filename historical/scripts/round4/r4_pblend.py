import pandas as pd, numpy as np, json
from sklearn.metrics import average_precision_score
A="/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
f5=pd.read_parquet(f"{A}/m5_both_train_oof.parquet")
v6=pd.read_parquet(f"{A}/m15_v6_drop_m26_train_oof.parquet")
v7=pd.read_parquet(f"{A}/m15_v7_drop_contrast_train_oof.parquet")
src="devsub11"
a=f5[f5.src==src].set_index("key").oof
b=v6[v6.src==src].set_index("key").oof
c=v7[v7.src==src].set_index("key").oof
lab=v6[v6.src==src].set_index("key")[["y","label","hid"]]
idx=a.index.intersection(b.index).intersection(c.index)
lab=lab.loc[idx]
ra=a.loc[idx].rank(pct=True); rb=b.loc[idx].rank(pct=True); rc=c.loc[idx].rank(pct=True)
res={}
def ev(score,name):
    y=lab.y.values
    m=(lab.label==1)|(lab.label==0)
    res[name]={"AP_clean":round(float(average_precision_score(y[m],score[m])),5),"AP_all":round(float(average_precision_score(y,score)),5)}
ev(a.loc[idx],"m5"); ev(b.loc[idx],"v6"); ev(c.loc[idx],"v7")
for w in [0.3,0.5,0.7]:
    ev((1-w)*ra+w*rb,f"blend_m5_v6_{w}")
    ev((1-w)*ra+w*rc,f"blend_m5_v7_{w}")
    ev((1-w)*rb+w*rc,f"blend_v6_v7_{w}")
    ev(ra/3+rb/3+rc/3,f"blend_all_{w}")
print(json.dumps(res,indent=1))
