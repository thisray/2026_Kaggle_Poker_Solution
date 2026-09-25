"""Evaluate TabICL CV with 24 vs 29 features (background-information features added), standalone and blended with the
r11 ranker blend (fixed w=0.6 as in r15, plus nested weight choice)."""
import numpy as np, pandas as pd, sys
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; T = f"{A_}/opus_r2_tabicl_20260918"
d = pd.read_parquet(f"{A_}/round11_scoped/dev_oof_aligned.parquet")[["slot", "hand_id", "fold", "ev", "m_p", "rs_blend"]]
r15tab = pd.read_csv(f"{A_}/round15_campaign/tabicl_cv/predictions.csv.gz")[["slot", "hand_id", "score"]].rename(columns={"score": "tab_r15"}); d = d.merge(r15tab, on=["slot", "hand_id"])
for k in sys.argv[1:]:
    t = pd.read_csv(f"{T}/cv_{k}/predictions.csv.gz")[["slot", "hand_id", "score"]].rename(columns={"score": f"tab{k}"}); d = d.merge(t, on=["slot", "hand_id"])
def ap5(g, col):
    top = g.sort_values(col, ascending=False, kind="mergesort").ev.values[:5]; hits = 0; s = 0.0
    for i, e in enumerate(top):
        if e: hits += 1; s += hits / (i + 1)
    return s / min(5, int(g.m_p.iloc[0]))
E = lambda c: d.groupby("slot").apply(lambda g: ap5(g, c))
R = lambda c: d.groupby("slot")[c].rank(pct=True)
d["r_rs"] = R("rs_blend")
meta = d.groupby("slot").fold.first()
for c in ["tab_r15"] + [f"tab{k}" for k in sys.argv[1:]]:
    d["r_c"] = R(c); e_alone = E(c)
    ws = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]; cache = {}
    for w in ws:
        d["tmp"] = w * d.r_c + (1 - w) * d.r_rs; cache[w] = E("tmp")
    tot = []; ch = []
    for f in range(5):
        trs = meta.index[meta != f]; tes = meta.index[meta == f]; b = max(ws, key=lambda w: cache[w].loc[trs].mean()); ch.append(b); tot += list(cache[b].loc[tes])
    print(f"{c:8s} alone {e_alone.mean():.6f} | blend w0.6 {cache[0.6].mean():.6f} | nested blend {np.mean(tot):.6f} (w {ch}) | per-fold w0.6 {[round(cache[0.6][meta == f].mean(), 4) for f in range(5)]}", flush=True)
