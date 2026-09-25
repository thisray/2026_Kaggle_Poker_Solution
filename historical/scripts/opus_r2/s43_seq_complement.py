"""Complementarity of round15 sequence views (action-MIL, GRU, action+background) with the current best E stack
(r15 = 0.6 TabICL + 0.4 r11 rank blend): error overlap + nested rank-blend weights (4 folds choose, 5th evaluates)."""
import numpy as np, pandas as pd, itertools
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; R15 = f"{A_}/round15_campaign"
d = pd.read_parquet(f"{A_}/round11_scoped/dev_oof_aligned.parquet")[["slot", "hand_id", "fold", "ev", "m_p", "rs_blend"]]
for nm, path in [("tab", "tabicl_cv"), ("act", "action_cv"), ("gru", "gru_cv"), ("actbg", "action_bg_cv")]:
    t = pd.read_csv(f"{R15}/{path}/predictions.csv.gz")[["slot", "hand_id", "score"]].rename(columns={"score": nm}); d = d.merge(t, on=["slot", "hand_id"])
assert len(d) == 7440, len(d)
R = lambda c: d.groupby("slot")[c].rank(pct=True)
for c in ["tab", "rs_blend", "act", "gru", "actbg"]: d["r_" + c] = R(c)
d["r15"] = 0.6 * d.r_tab + 0.4 * d.r_rs_blend; d["r_r15"] = R("r15")
def ap5(g, col):
    top = g.sort_values(col, ascending=False, kind="mergesort").ev.values[:5]; hits = 0; s = 0.0
    for i, e in enumerate(top):
        if e: hits += 1; s += hits / (i + 1)
    return s / min(5, int(g.m_p.iloc[0]))
def per_pair(col): return d.groupby("slot").apply(lambda g: ap5(g, col))
base = per_pair("r15"); print("E r15", round(base.mean(), 6), " act", round(per_pair("act").mean(), 4), " gru", round(per_pair("gru").mean(), 4), " actbg", round(per_pair("actbg").mean(), 4))
# error overlap: evidence hands missed by r15 top-5 that a sequence view ranks in its top-5
d["r15_rank"] = d.groupby("slot").r15.rank(ascending=False, method="first")
for c in ["act", "gru", "actbg"]:
    d["rk"] = d.groupby("slot")[c].rank(ascending=False, method="first")
    miss = d[(d.ev == 1) & (d.r15_rank > 5)]; hit = d[(d.ev == 1) & (d.r15_rank <= 5)]
    print(f"{c}: of r15-missed evidence ({len(miss)}), share in {c} top-5: {(miss.rk <= 5).mean():.3f};  of r15-hit evidence, share in {c} top-5: {(hit.rk <= 5).mean():.3f}")
meta = d.groupby("slot").fold.first()
for c in ["act", "gru", "actbg"]:
    ws = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]; cache = {}
    for w in ws:
        d["tmp"] = (1 - w) * d.r_r15 + w * d["r_" + c]; cache[w] = per_pair("tmp")
    tot = []; ch = []
    for f in range(5):
        trs = meta.index[meta != f]; tes = meta.index[meta == f]
        best = max(ws, key=lambda w: cache[w].loc[trs].mean()); ch.append(best); tot += list(cache[best].loc[tes])
    print(f"nested r15 + {c}: E {np.mean(tot):.6f} (delta {np.mean(tot) - base.mean():+.6f}); chosen weights {ch}; optimistic best {max(ws, key=lambda w: cache[w].mean())} -> {max(cache[w].mean() for w in ws) - base.mean():+.6f}")
