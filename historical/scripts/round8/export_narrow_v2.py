import numpy as np, pandas as pd
OPUS="/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
RAW="/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
DST="/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917"
n=pd.read_csv(f"{DST}/r6_narrow_candidates.csv")
lab=pd.read_csv(f"{RAW}/development_labels.csv")
lab["lo"]=lab[["player_1","player_2"]].min(axis=1); lab["hi"]=lab[["player_1","player_2"]].max(axis=1)
fam=lab.set_index(["lo","hi"]).behavior_family
n["family"]=n.set_index(["pair_player_lo","pair_player_hi"]).index.map(fam)
# timestamps: hand_id -> h_ts (sorted by time), plus pair-relative percentile
hidx=pd.read_parquet(f"{OPUS}/np/hand_index.parquet").set_index("hand_id")
ts=np.load(f"{OPUS}/np/h_ts.npy")
n["hand_ts"]=hidx.hi.reindex(n.hand_id).map(lambda i: ts[int(i)]).values
g=n.groupby("slot").hand_ts
n["ts_pct_in_pair"]=g.rank(pct=True).values
n["ts_rank_in_pair"]=g.rank(method="first").values
n.to_csv(f"{DST}/r6_narrow_candidates_v2.csv", index=False)
print("rows", len(n), "cols", n.shape[1])
print("family counts", n.family.value_counts(dropna=False).to_dict())
print("ts_pct quantiles", np.round(np.quantile(n.ts_pct_in_pair,[0.05,0.25,0.5,0.75,0.95]),3))
