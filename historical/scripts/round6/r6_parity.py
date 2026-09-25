import pandas as pd, numpy as np, json
from scipy.stats import poisson
DST="/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917"
A="/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
S=pd.read_parquet(f"{DST}/r6_dev_scores.parquet").sort_values(["sl","ts"]).reset_index(drop=True)
# emulate deployment: lin only for top-20 candidates by u0
def ap5(g,n_g):
    hits=0;s=0.0
    for r,z in enumerate(g[:5],start=1):
        if z: hits+=1; s+=hits/r
    return s/min(5,max(int(n_g),1))
import numpy as np
# lin values were not saved; recompute proxy via u_rr - u_r5 (both saved)
S["lin05"]=S.u_rr.values-S.u_r5.values   # 0.5*lin
S["nnterm"]=S.u_r5b.values-S.u_rr.values # 0.1*nn
res={}
# research path (lin for all rows)
res["research_E"]=round(float(np.mean([ap5(g.ev.values[np.argsort(-g.u_r5b.values)], int(g.ev.sum())) for _,g in S.groupby("sl")])),4)
# deployment path: zero lin outside top-20 by u0
dep=S.copy()
top20=dep.sort_values(["sl","u_r5"],ascending=[True,False]).groupby("sl").head(20).index
mask=np.zeros(len(dep),bool); mask[top20]=True
dep["lin05d"]=np.where(mask,dep.lin05.values,0.0)
dep["ud"]=dep.u_r5.values+dep.lin05d+dep.nnterm.values
res["deploy_E"]=round(float(np.mean([ap5(g.ev.values[np.argsort(-g.ud.values)], int(g.ev.sum())) for _,g in dep.groupby("sl")])),4)
# top5 set comparison
a=S.sort_values(["sl","u_r5b"],ascending=[True,False]).groupby("sl").head(5)[["sl","h"]]
b=dep.sort_values(["sl","ud"],ascending=[True,False]).groupby("sl").head(5)[["sl","h"]]
inter=sum(len(set(x.h)&set(y.h)) for x,y in zip([g for _,g in a.groupby("sl")],[g for _,g in b.groupby("sl")]))
res["top5_intersection_of_1860"]=int(inter)
res["pairs_with_any_diff"]=int(sum(1 for x,y in zip([g for _,g in a.groupby("sl")],[g for _,g in b.groupby("sl")]) if set(x.h)!=set(y.h)))
print(json.dumps(res,indent=2))
json.dump(res,open(f"{DST}/r6_parity.json","w"),indent=2)
