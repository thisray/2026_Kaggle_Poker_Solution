"""Top-5 false positives of the r11 blend: in-window (true non-collusion) vs post-window (possibly later collusion hands)."""
import numpy as np, pandas as pd
A = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A}/opus_r1_20260917"
d = pd.read_parquet(f"{A}/round11_scoped/dev_oof_aligned.parquet")
n = pd.read_csv(f"{A}/round3_research_20260917/r6_narrow_candidates_v2.csv", usecols=["slot", "hand_id", "family", "hand_ts", "ts_rank_in_pair"])
d = d.merge(n, on=["slot", "hand_id"], how="left")
hidx = pd.read_parquet(f"{OUT}/np/hand_index.parquet"); hmap = dict(zip(hidx.hand_id, hidx.hi)); d["h"] = d.hand_id.map(hmap).astype(np.int64)
M = pd.read_parquet(f"{OUT}/m25e1_handfeat2_m19w10_oof.parquet")[["sl", "h", "s", "sc_fam"]].rename(columns={"sl": "slot"})
d = d.merge(M, on=["slot", "h"], how="left"); print("m19 score coverage", d.s.notna().mean())
last = d[d.ev == 1].groupby("slot").hand_ts.max(); d["post"] = d.hand_ts > d.slot.map(last)
d["r"] = d.groupby("slot").rs_blend.rank(ascending=False, method="first")
top = d[d.r <= 5]; fp = top[top.ev == 0]
print(f"top5 picks {len(top)}; FPs {len(fp)}; FP post-window share {fp.post.mean():.3f}; all candidates post-window share {d.post.mean():.3f}")
print("FP by family post share:", fp.groupby("family").post.mean().round(3).to_dict())
miss = d[(d.ev == 1) & (d.r > 5)]
d["tr"] = d.groupby("slot").hand_ts.rank()
ev = d[d.ev == 1].copy(); ev["ev_order"] = ev.groupby("slot").hand_ts.rank()
print("missed evidence by evidence-time order (1=earliest):", ev[ev.r > 5].ev_order.value_counts().sort_index().to_dict(), " of", ev.ev_order.value_counts().sort_index().to_dict())
# the m19 (time-agnostic collusion detector) score of FPs: post-window FPs vs in-window FPs vs evidence
for nm, g in [("evidence", d[d.ev == 1]), ("FP in-window", fp[~fp.post]), ("FP post-window", fp[fp.post]), ("non-top5 post", d[(d.r > 5) & d.post & (d.ev == 0)]), ("non-top5 in-win nonev", d[(d.r > 5) & ~d.post & (d.ev == 0)])]:
    print(f"{nm:24s} n {len(g):5d}  m19 s median {g.s.median():.3f}  mean {g.s.mean():.3f}  frac s>0.5 {(g.s > 0.5).mean():.3f}")
