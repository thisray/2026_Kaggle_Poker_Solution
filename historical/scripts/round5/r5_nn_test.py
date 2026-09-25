import pandas as pd, numpy as np, json
from scipy.stats import poisson
A="/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
F=pd.read_parquet(f"{A}/m25famsep4_oof.parquet").sort_values(["sl","ts"]).reset_index(drop=True)
S=pd.read_parquet(f"{A}/seqwithin_oof.parquet").sort_values(["sl","ts"]).reset_index(drop=True)
assert (S.sl.values==F.sl.values).all()
cum=F.groupby("sl").sc.cumsum()-F.sc
F["u0"]=np.log(np.clip(F.sc*poisson.cdf(3,cum)*np.exp(-0.25*F.groupby("sl").ts.rank(pct=True)),1e-9,None))
def map5(df,col):
    aps=[]
    for _,g in df.groupby("sl"):
        rel=set(g.h[g.ev]); g=g.sort_values(col,ascending=False)
        hits=0; s=0.0
        for i,hh in enumerate(g.h.values[:5]):
            if hh in rel: hits+=1; s+=hits/(i+1)
        aps.append(s/min(5,max(len(rel),1)))
    return round(float(np.mean(aps)),4)
res={"famsep":map5(F,"u0")}
lg=lambda p: np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
for a in [0.1,0.25,0.5,1.0]:
    F["u"]=F.u0.values+a*lg(S.nn_cal.values)
    res[f"nn_a{a}"]=map5(F,"u")
with open("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917/r5_nn_test.json","w") as f: json.dump(res,f,indent=2)
print(json.dumps(res,indent=1))
