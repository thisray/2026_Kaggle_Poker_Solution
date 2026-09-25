import pandas as pd, numpy as np, json
from sklearn.metrics import average_precision_score
A="/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
res={}
files={"m5":"m5_both_train_oof.parquet","v6drop":"m15_v6_drop_m26_train_oof.parquet","v7":"m15_v7_drop_contrast_train_oof.parquet"}
D={k:pd.read_parquet(f"{A}/{f}") for k,f in files.items()}
for src in ["devsub11","devsub12"]:
    ref=D["v6drop"]; ref=ref[ref.src==src][["key","y","label","hid"]].drop_duplicates("key").set_index("key")
    for k in D:
        s=D[k]; s=s[s.src==src].drop_duplicates("key").set_index("key").oof
        idx=ref.index.intersection(s.index)
        y=ref.loc[idx,"y"].values; lab=ref.loc[idx,"label"].values
        m=(lab==1)|(lab==0)
        res[f"{src}_{k}"]={"AP_clean":round(float(average_precision_score(y[m],s.loc[idx][m])),5),"AP_all":round(float(average_precision_score(y,s.loc[idx])),5),
                           "top450_pos":int(pd.Series(y).sort_values(ascending=False).head(0).sum())}
        t=pd.DataFrame({"y":y,"s":s.loc[idx]}).sort_values("s",ascending=False).head(450)
        res[f"{src}_{k}"]["top450_pos"]=int(t.y.sum())
        res[f"{src}_{k}"]["top450_hidden"]=int(ref.loc[t.index,"hid"].sum())
        res[f"{src}_{k}"]["top450_neg"]=int((ref.loc[t.index,"label"]==0).sum())
print(json.dumps(res,indent=1))
