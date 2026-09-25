import pandas as pd, numpy as np, json
A="/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
F=pd.read_parquet(f"{A}/m25famsep4_oof.parquet").sort_values(["sl","ts"])
def ap5(g, col):
    rel=set(g.h[g.ev]); g=g.sort_values(col,ascending=False)
    hits=0; s=0.0
    for i,hh in enumerate(g.h.values[:5]):
        if hh in rel: hits+=1; s+=hits/(i+1)
    return s/min(5,max(len(rel),1))
# joint reference from t1 (same fold structure)
T=pd.read_parquet(f"{A}/m25t1_handfeat2_m19w10_oof.parquet").sort_values(["sl","ts"])
from scipy.stats import poisson
def dec(D):
    cum=D.groupby("sl").sc_fam.cumsum()-D.sc_fam if "sc_fam" in D.columns else None
    return cum
T["y"]=T.sc_fam*poisson.cdf(3,T.groupby("sl").sc_fam.cumsum()-T.sc_fam)*np.exp(-0.25*T.groupby("sl").ts.rank(pct=True))
F["y"]=F.sc*poisson.cdf(3,F.groupby("sl").sc.cumsum()-F.sc)*np.exp(-0.25*F.groupby("sl").ts.rank(pct=True))
aj=T.groupby("sl").apply(lambda g: ap5(g,"y"))
af=F.groupby("sl").apply(lambda g: ap5(g,"y"))
d=(af-aj)
print("pairs",len(d),"mean delta",round(float(d.mean()),4),"median",round(float(d.median()),4))
print("improved", int((d>0.01).sum()), "worsened", int((d<-0.01).sum()), "same", int((d.abs()<=0.01).sum()))
rng=np.random.RandomState(7); boots=[]
for b in range(2000):
    idx=rng.randint(0,len(d),len(d)); boots.append(d.values[idx].mean())
print("bootstrap 95% CI:", [round(float(np.quantile(boots,0.025)),4), round(float(np.quantile(boots,0.975)),4)])
json.dump({"mean_delta":round(float(d.mean()),4),"improved":int((d>0.01).sum()),"worsened":int((d<-0.01).sum()),
           "ci95":[round(float(np.quantile(boots,0.025)),4),round(float(np.quantile(boots,0.975)),4)]},
          open("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917/r4_fam_boot.json","w"),indent=2)
