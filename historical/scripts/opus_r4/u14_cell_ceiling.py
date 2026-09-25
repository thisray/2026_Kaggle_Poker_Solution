"""R4-U14: inside the most common DT evidence cell (small pot, R wins, S passive, R aggressive postflop, S folds), how much do the cards / time / R15 separate listed vs unlisted in-window hands?"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
pd.set_option("display.width", 250)
m = pd.read_parquet(f"{OUT}/r4/u5_directed_transfer.parquet"); f = pd.read_parquet(f"{OUT}/r3/t58_seq_feats.parquet").rename(columns={"slot": "sl"}).drop(columns=["pa", "pb"]); m = m.merge(f, on=["sl", "h"])
HS1 = np.load(f"{OUT}/HS1.npy", mmap_mode="r")
rs = np.where(m.recvA, m.sa, m.sb).astype(int); ss = np.where(m.recvA, m.sb, m.sa).astype(int)
m["hsS_pre"] = [float(HS1[h, 0, s]) for h, s in zip(m.h.values, ss)]; m["hsR_pre"] = [float(HS1[h, 0, s]) for h, s in zip(m.h.values, rs)]
m["hsS_last"] = [float(HS1[h, int(st), s]) for h, s, st in zip(m.h.values, ss, m.stmax.values)]
w = m[(m.zone != "post")]
cell = w[(~w.big) & (w.conR > 1) & (w.conS > 1) & (w.dir == 1) & (~w.s_aggr) & (w.s_last == 0) & w.r_aggr_post]
print("cell rows", len(cell), "ev rate", cell.ev.mean(), "pairs", cell.sl.nunique())
cell = cell.copy(); cell["hsb"] = pd.qcut(cell.hsS_pre, 8, labels=False, duplicates="drop"); print("ev rate by S preflop strength octile"); print(cell.groupby("hsb").ev.agg(["mean", "size"]).round(3).T.to_string())
cell["hlb"] = pd.qcut(cell.hsS_last, 8, labels=False, duplicates="drop"); print("ev rate by S strength on the street where it folded"); print(cell.groupby("hlb").ev.agg(["mean", "size"]).round(3).T.to_string())
cell["kb"] = pd.qcut(cell.k, 6, labels=False, duplicates="drop"); print("ev rate by co-seated index"); print(cell.groupby("kb").ev.agg(["mean", "size"]).round(3).T.to_string())
for c in ["hsS_pre", "hsS_last", "hsR_pre", "k", "sur_pass_s_max", "sur_all_sum", "str_fold_max", "mismatch_max", "own", "par"]:
    print(f"{c:18s} AUC {roc_auc_score(cell.ev, cell[c]):.3f}")
cc = cell[cell.b.notna()]; print("R15 blend AUC (candidates only, cover", round(len(cc) / len(cell), 3), ")", round(roc_auc_score(cc.ev, cc.b), 3), " ev rate among non-candidates", cell[cell.b.isna()].ev.mean().round(3))
cc = cc.copy(); cc["top5"] = cc.rk <= 5; print(cc.groupby("top5").ev.agg(["mean", "size"]))
