"""Family-specific decoding: per family, nested choice among r11, time-agnostic detector + exact first-K decoder, and rank blends."""
import numpy as np, pandas as pd, itertools, json
A = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"
m = pd.read_parquet(f"{A}/opus_r1_20260917/s12_coll_decoder.parquet")
fam = pd.read_csv(f"{A}/round3_research_20260917/r6_narrow_candidates_v2.csv", usecols=["slot", "family"]).drop_duplicates("slot")
m = m.merge(fam, on="slot")
def ap5(g, col):
    top = g.sort_values(col, ascending=False, kind="mergesort").ev.values[:5]; hits = 0; s = 0.0
    for i, e in enumerate(top):
        if e: hits += 1; s += hits / (i + 1)
    return s / min(5, int(g.m_p.iloc[0]))
per = {}
m["rn"] = m.groupby("slot").rs_blend.rank(pct=True)
m["tp"] = m.groupby("slot").hand_ts.rank(pct=True)
for K in [3, 4, 5, 6, 7]: m[f"dn{K}"] = m.groupby("slot")[f"dec{K}"].rank(pct=True)
m["pn"] = m.groupby("slot").p_raw.rank(pct=True)
cands = {}
for K in [3, 4, 5, 6, 7]:
    for w in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 1e3]:
        cands[("dec", K, w)] = m.rn + w * m[f"dn{K}"]
for a in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]:
    for w in [0.5, 1.0, 2.0, 1e3]:
        cands[("praw_time", a, w)] = m.rn + w * (m.pn - a * m.tp)
cands[("base",)] = m.rs_blend
keys = list(cands)
tab = {}
for k in keys:
    m["tmp"] = cands[k]
    tab[k] = m.groupby("slot").apply(lambda g: ap5(g, "tmp")).rename("ap")
slot_meta = m.groupby("slot").agg(fold=("fold", "first"), family=("family", "first"))
res = {}
tot_new = []; tot_base = []
for fm in ["coordinated_isolation", "directed_transfer", "soft_play"]:
    sm = slot_meta[slot_meta.family == fm]
    new = []
    for f in range(5):
        trs = sm.index[sm.fold != f]; tes = sm.index[sm.fold == f]
        best = max(keys, key=lambda k: tab[k].loc[trs].mean())
        new.append((f, best, float(tab[best].loc[tes].mean() - tab[("base",)].loc[tes].mean()), len(tes)))
        tot_new.extend(tab[best].loc[tes].values); tot_base.extend(tab[("base",)].loc[tes].values)
    d = sum(x[2] * x[3] for x in new) / sum(x[3] for x in new)
    opt = max(keys, key=lambda k: tab[k].loc[sm.index].mean())
    print(fm[:2], "nested delta", round(d, 5), [(x[1], round(x[2], 4)) for x in new], "| optimistic", opt, round(tab[opt].loc[sm.index].mean() - tab[("base",)].loc[sm.index].mean(), 5), flush=True)
print("overall nested E:", round(np.mean(tot_new), 6), "base", round(np.mean(tot_base), 6), "delta", round(np.mean(tot_new) - np.mean(tot_base), 6))
