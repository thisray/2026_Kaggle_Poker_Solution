"""Relative vs absolute time: dev (OOF top-5 picks of the deployed-style blend + true evidence) vs eval (deployed r2j2m picks)
in absolute co-seated hand index k and relative percentile, for known-family pairs.  If eval picks are much earlier in k than
dev evidence / dev picks, the evidence models carry a relative-time prior that is miscalibrated for the shorter eval period."""
import numpy as np, pandas as pd
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"
ts = np.load(f"{D}/h_ts.npy")
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); id2hi = dict(zip(hidx.hand_id, hidx.hi))
# dev: all co-seated dev hands of the 372 positive pairs (s66 table has every co-seated hand) + s59 picks
M = pd.read_parquet(f"{OUT}/s66_dev_outcomes.parquet")[["sl", "h", "ev", "fam", "ts"]].sort_values(["sl", "ts"])
M["k"] = M.groupby("sl").cumcount(); M["n"] = M.groupby("sl").sl.transform("size"); M["pct"] = (M.k + 0.5) / M.n
R = pd.read_parquet(f"{OUT}/s59_candidates_with_channels.parquet"); R["h"] = R.hand_id.map(id2hi)
R = R.sort_values(["slot", "rs_blend"], ascending=[True, False]); R["r"] = R.groupby("slot").cumcount()
pick = R[R.r < 5][["slot", "h"]].rename(columns={"slot": "sl"}).merge(M[["sl", "h", "k", "pct", "n"]], on=["sl", "h"], how="left")
ev = M[M.ev]
print("DEV  co-seated hands per positive pair: median", int(M.groupby("sl").size().median()))
print("DEV  evidence   k quantiles", np.round(ev.k.quantile([.1, .25, .5, .75, .9]).values, 1), " pct", np.round(ev.pct.quantile([.25, .5, .75, .9]).values, 3))
print("DEV  top-5 picks k quantiles", np.round(pick.k.quantile([.1, .25, .5, .75, .9]).values, 1), " pct", np.round(pick.pct.quantile([.25, .5, .75, .9]).values, 3))
# eval: deployed picks for known-family predicted pairs in the top 400
b = pd.read_csv(f"{C}/r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv", dtype=str)
b["risk"] = b.risk_score.astype(float); b = b.sort_values("risk", ascending=False).head(400)
b = b[b.predicted_behavior.isin(["directed_transfer", "soft_play", "coordinated_isolation"])]
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet"); pid2sl = c38.set_index("pair_id").slot
slots = pid2sl.loc[b.pair_id].values
H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, slots)
E = pd.DataFrame({"sl": SL[m], "h": H[m]}); E["ts"] = ts[E.h]; E = E.sort_values(["sl", "ts"]); E["k"] = E.groupby("sl").cumcount(); E["n"] = E.groupby("sl").sl.transform("size"); E["pct"] = (E.k + 0.5) / E.n
rows = []
for r in b.itertuples():
    for c in [f"evidence_hand_{i}" for i in range(1, 6)]:
        h = id2hi.get(getattr(r, c), -1)
        if h >= 0: rows.append((pid2sl[r.pair_id], h))
P = pd.DataFrame(rows, columns=["sl", "h"]).merge(E, on=["sl", "h"], how="left")
print("EVAL co-seated hands per top pair: median", int(E.groupby("sl").size().median()))
print("EVAL top-5 picks k quantiles", np.round(P.k.quantile([.1, .25, .5, .75, .9]).values, 1), " pct", np.round(P.pct.quantile([.25, .5, .75, .9]).values, 3))
