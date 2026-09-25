import pandas as pd, numpy as np, json
from scipy.stats import poisson
A="/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
F=pd.read_parquet(f"{A}/m25famsep4_oof.parquet").sort_values(["sl","ts"]).reset_index(drop=True)
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
lg=lambda p: np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
res={"base":map5(F,"u0")}
views={"t1":"m25t1_handfeat2_m19w10_oof.parquet","p2":"m25p2_handfeat2_m19w10_oof.parquet","e5":"m25e5_handfeat5_m19w10_oof.parquet","m21a":"m25_handfeat2_m21a_oof.parquet"}
for nm,f in views.items():
    d=pd.read_parquet(f"{A}/{f}").sort_values(["sl","ts"]).reset_index(drop=True)
    assert (d.sl.values==F.sl.values).all()
    v=lg(d.sc_fam.values)
    for a in [0.1,0.25]:
        F["u"]=F.u0.values+a*v
        res[f"{nm}_a{a}"]=map5(F,"u")
S=pd.read_parquet(f"{A}/seqwithin_oof.parquet").sort_values(["sl","ts"]).reset_index(drop=True)
nn=lg(S.nn_cal.values)
for a in [0.05,0.1,0.15,0.2]:
    F["u"]=F.u0.values+a*nn
    res[f"nn_a{a}"]=map5(F,"u")
# nn + t1 combo
for a,b in [(0.1,0.1),(0.15,0.1),(0.1,0.2)]:
    F["u"]=F.u0.values+a*nn+b*lg(pd.read_parquet(f"{A}/m25t1_handfeat2_m19w10_oof.parquet").sort_values(["sl","ts"]).reset_index(drop=True).sc_fam.values)
    res[f"nn{a}_t1_{b}"]=map5(F,"u")
with open("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917/r5_views_add.json","w") as f: json.dump(res,f,indent=2)
print(json.dumps(res,indent=1))
