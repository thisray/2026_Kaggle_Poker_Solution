"""R4 probe: compare pair-risk OOF variants (m5_both submitted vs m15_v7 latest) on dev population."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

A = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917"
rep = {}


def load(f):
    d = pd.read_parquet(f"{A}/{f}")
    return d


def metrics(d, name):
    d = d.drop_duplicates("key").copy()
    r = {}
    for src in ["devsub11", "devsub12"]:
        s = d[d.src == src]
        if len(s) == 0:
            continue
        y = s.y.values
        # clean: positives vs confirmed negatives only
        m_clean = (s.label == 1) | (s.label == 0)
        r[f"{src}_AP_clean"] = round(float(average_precision_score(y[m_clean], s.oof[m_clean])), 5)
        r[f"{src}_AP_all"] = round(float(average_precision_score(y, s.oof)), 5)
        top = s.sort_values("oof", ascending=False).head(450)
        r[f"{src}_top450_pos"] = int(top.y.sum())
        hid = s.hid.values if "hid" in s.columns else np.zeros(len(s), bool)
        r[f"{src}_top450_hidden"] = int(top.hid.sum()) if "hid" in s.columns else None
        r[f"{src}_top450_neg"] = int((top.label == 0).sum())
    rep[name] = r
    return r


m5 = load("m5_both_train_oof.parquet")
print("m5 cols:", list(m5.columns), len(m5))
metrics(m5, "m5_both")
v7 = load("m15_v7_drop_contrast_train_oof.parquet")
print("v7 cols:", list(v7.columns), len(v7))
metrics(v7, "m15_v7_drop_contrast")
v6d = load("m15_v6_drop_m26_train_oof.parquet")
metrics(v6d, "m15_v6_drop_m26")
v6n = load("m15_v6_none_m26_train_oof.parquet")
metrics(v6n, "m15_v6_none_m26")

# score distribution agreement and rank blend potential
a = m5.drop_duplicates("key").set_index("key").oof
b = v7.drop_duplicates("key").set_index("key").oof
common = a.index.intersection(b.index)
from scipy.stats import spearmanr
rep["spearman_m5_vs_v7"] = round(float(spearmanr(a[common], b[common]).correlation), 4)
with open(f"{OUT}/r4_pline_probe.json", "w") as f:
    json.dump(rep, f, indent=2)
print(json.dumps(rep, indent=2))
