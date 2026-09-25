"""E error anatomy on the r11 blend OOF: per family, per |G|, per pair exposure; where are the lost AP points."""
import numpy as np, pandas as pd
A = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"
d = pd.read_parquet(f"{A}/round11_scoped/dev_oof_aligned.parquet")
n = pd.read_csv(f"{A}/round3_research_20260917/r6_narrow_candidates_v2.csv", usecols=["slot", "hand_id", "family", "hand_ts", "ts_rank_in_pair", "h_p"])
d = d.merge(n, on=["slot", "hand_id"], how="left")
def ap5(g, col="rs_blend"):
    top = g.sort_values(col, ascending=False, kind="mergesort").ev.values[:5]; hits = 0; s = 0.0
    for i, e in enumerate(top):
        if e: hits += 1; s += hits / (i + 1)
    return s / min(5, int(g.m_p.iloc[0]))
rows = []
for sl, g in d.groupby("slot"):
    g = g.sort_values("rs_blend", ascending=False)
    evr = np.flatnonzero(g.ev.values) + 1
    rows.append(dict(slot=sl, fam=g.family.iloc[0], m_p=int(g.m_p.iloc[0]), n_ev_in_cand=int(g.ev.sum()), ap=ap5(g), h_p=g.h_p.iloc[0],
                     hits5=int(g.ev.values[:5].sum()), worst_ev_rank=int(evr.max()) if len(evr) else 99, ev_ranks=tuple(evr)))
P = pd.DataFrame(rows)
print("E", round(P.ap.mean(), 6), "pairs", len(P))
print(P.groupby("fam").agg(E=("ap", "mean"), n=("ap", "size"), hits5=("hits5", "mean"), m_p=("m_p", "mean")).round(4))
print(P.groupby("m_p").agg(E=("ap", "mean"), n=("ap", "size")).round(4))
print("h_p values", P.h_p.value_counts().to_dict())
lost = (1 - P.ap).sum(); print("total lost AP mass", round(lost, 2))
P["recall_loss"] = (P.m_p - P.hits5) / P.m_p
print("mean hits@5 / m_p:", round((P.hits5 / P.m_p.clip(upper=5)).mean(), 4), " pairs with all evidence in top5:", int((P.hits5 == P.m_p.clip(upper=5)).sum()))
# decompose: loss from missing evidence vs loss from ordering
def ap_perfect_order(r):
    k = r.hits5; m = min(5, r.m_p); return k / m
P["ap_order_oracle"] = P.apply(ap_perfect_order, axis=1)
print("E if same top-5 set but perfect order:", round(P.ap_order_oracle.mean(), 4))
print("distribution of hits5:", P.hits5.value_counts().sort_index().to_dict())
print("ev-rank histogram (all ev in candidates):", pd.Series([r for t in P.ev_ranks for r in t]).value_counts().sort_index().to_dict())
