import numpy as np, pandas as pd, json
DST="/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917"
R8="/home/thisray/projects/260916_Kaggle_Poker_artifacts/round8_raw_20260917"
def ap5(o,m):
    hits=0;s=0.0
    for r,e in enumerate(o[:5],start=1):
        if e: hits+=1; s+=hits/r
    return s/min(5,max(int(m),1))
n=pd.read_csv(f"{DST}/r6_narrow_candidates_v2.csv")
base=[]
for slot,g in n.groupby("slot"):
    g=g.sort_values("u_r5b",ascending=False)
    base.append({"slot":slot,"E_base":ap5(g.ev.values.astype(int),g.m_p.iloc[0])})
b=pd.DataFrame(base)
pp=pd.read_csv(f"{R8}/moments_plus_seed71/per_pair.csv").rename(columns={"E":"E_res"})
d=b.merge(pp[["slot","E_res"]],on="slot")
fam=n.groupby("slot").family.first(); tsc=n.groupby("slot").hand_ts.mean()
d["family"]=d.slot.map(fam); d["mean_ts"]=d.slot.map(tsc)
d["ts_stratum"]=pd.qcut(d.mean_ts,3,labels=["early","mid","late"])
def summ(g):
    return pd.Series({"pairs":len(g),"E_base":round(g.E_base.mean(),4),"E_res":round(g.E_res.mean(),4),"delta":round((g.E_res-g.E_base).mean(),4)})
res={"overall":summ(d).to_dict()}
for key in ["family","ts_stratum"]:
    res[key]=d.groupby(key).apply(summ).round(4).to_dict("index")
print(json.dumps(res,indent=2))
json.dump(res,open(f"{DST}/r8_strata.json","w"),indent=2)
