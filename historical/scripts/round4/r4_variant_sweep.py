import pandas as pd, numpy as np, os
from scipy.stats import poisson
A="/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
files=[f for f in os.listdir(A) if f.startswith("m25") and f.endswith("_oof.parquet")]
def map5(df,col):
    aps=[]
    for sl,g in df.groupby("sl"):
        rel=set(g.h[g.ev]); g=g.sort_values(col,ascending=False)
        hits=0; s=0.0
        for i,hh in enumerate(g.h.values[:5]):
            if hh in rel: hits+=1; s+=hits/(i+1)
        aps.append(s/min(5,max(len(rel),1)))
    return round(float(np.mean(aps)),4)
rows=[]
for f in sorted(files):
    D=pd.read_parquet(f"{A}/{f}").sort_values(["sl","ts"])
    c="sc_fam" if "sc_fam" in D.columns else D.columns[-1]
    D["cum"]=D.groupby("sl")[c].cumsum()-D[c]
    D["x4"]=D[c]*poisson.cdf(4,D["cum"])
    D["x3"]=D[c]*poisson.cdf(3,D["cum"])*np.exp(-0.25*D.groupby("sl").ts.rank(pct=True))
    rows.append((f.replace("_oof.parquet",""), map5(D,c), map5(D,"x4"), map5(D,"x3")))
    print(rows[-1], flush=True)
