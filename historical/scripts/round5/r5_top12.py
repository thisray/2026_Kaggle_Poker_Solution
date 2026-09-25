import pandas as pd, numpy as np, json
from scipy.stats import poisson
A="/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
F=pd.read_parquet(f"{A}/m25famsep4_oof.parquet").sort_values(["sl","ts"]).reset_index(drop=True)
def dec(col):
    cum=F.groupby("sl")[col].cumsum()-F[col]
    return F[col]*poisson.cdf(3,cum)*np.exp(-0.25*F.groupby("sl").ts.rank(pct=True))
F["dec"]=dec("sc")
def map5(df,col):
    aps=[]
    for _,g in df.groupby("sl"):
        rel=set(g.h[g.ev]); g=g.sort_values(col,ascending=False)
        hits=0; s=0.0
        for i,hh in enumerate(g.h.values[:5]):
            if hh in rel: hits+=1; s+=hits/(i+1)
        aps.append(s/min(5,max(len(rel),1)))
    return round(float(np.mean(aps)),4)
res={}
res["raw_top5"]=map5(F,"sc")
res["dec_top5"]=map5(F,"dec")
# rank of evidence hands by decoded score
F["rk_dec"]=F.groupby("sl")["dec"].rank(ascending=False,method="first")
ev=F[F.ev]
res["evidence_rank_hist"]=ev.rk_dec.value_counts().sort_index().head(15).to_dict()
res["evidence_in_top12"]=int((ev.rk_dec<=12).sum())
res["evidence_total"]=int(len(ev))
# among top-12 rows: how many are evidence per rank
top12=F[F.rk_dec<=12]
res["top12_precision"]=round(float(top12.ev.mean()),4)
# try alternate selection: top5 by dec within time-first-ordered? and "one per episode"? 
# simple variant: rank by dec but require distinct street? no. Try: rank by sc (no time prior)
F["rk_sc"]=F.groupby("sl")["sc"].rank(ascending=False,method="first")
res["evidence_in_top12_by_sc"]=int((F[F.ev].rk_sc<=12).sum())
# dec without time factor: K3 only
cum=F.groupby("sl").sc.cumsum()-F.sc
F["dec_notime"]=F.sc*poisson.cdf(3,cum)
res["notime_top5"]=map5(F,"dec_notime")
# take best 5 of top-12 by dec_notime? (same as notime_top5)
# variant: weighted blend of dec and diversity: penalize same street? skip
# per-family raw vs dec
for fam in ["directed_transfer","soft_play","coordinated_isolation"]:
    G=F[F.fam==fam]
    res[f"{fam}_dec"]=map5(G,"dec")
# rank distribution of MISSED evidence (rank 6..)
miss=ev[ev.rk_dec>5]
res["missed_rank_hist"]=miss.rk_dec.value_counts().sort_index().head(12).to_dict()
print(json.dumps(res,indent=1))
json.dump(res,open("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917/r5_top12.json","w"),indent=2)
