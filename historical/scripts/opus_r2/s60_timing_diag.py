"""DIAGNOSTIC ONLY (rules question, not deployed): does hand duration (gap to the next hand at the same table) or its
per-action normalisation differ between evidence and non-evidence candidate hands, controlling for #actions?"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; RAW = f"{A_}/data/raw"
H = pd.read_parquet(f"{RAW}/hands.parquet", columns=["hand_id", "table_id", "started_at", "phase"])
H = H.sort_values(["table_id", "started_at"]).reset_index(drop=True)
H["t"] = H.started_at.astype("int64") / 1e9
H["dur"] = H.groupby("table_id").t.shift(-1) - H.t
nact = pd.read_parquet(f"{RAW}/actions.parquet", columns=["hand_id"]).hand_id.value_counts().rename("nact")
H = H.merge(nact, left_on="hand_id", right_index=True, how="left")
print("duration quantiles (s):", H.dur.quantile([0.01, 0.1, 0.5, 0.9, 0.99]).round(2).to_dict())
H["dur_per_act"] = H.dur / H.nact.clip(lower=1)
# residual duration given number of actions (median by nact)
med = H.groupby("nact").dur.median(); H["dur_res"] = H.dur - H.nact.map(med)
d = pd.read_parquet(f"{A_}/round11_scoped/dev_oof_aligned.parquet")[["slot", "hand_id", "ev"]]
d = d.merge(H[["hand_id", "dur", "nact", "dur_per_act", "dur_res"]], on="hand_id", how="left")
print("candidates", len(d), " missing dur", int(d.dur.isna().sum()))
d = d.dropna()
for c in ["dur", "nact", "dur_per_act", "dur_res"]:
    print(f"AUC ev vs non-ev candidates, {c}: {roc_auc_score(d.ev, d[c]):.3f}   mean ev {d[d.ev == 1][c].mean():.3f} non-ev {d[d.ev == 0][c].mean():.3f}")
# within-pair: does residual duration rank evidence?
d["r"] = d.groupby("slot").dur_res.rank(pct=True); print("within-pair AUC of dur_res rank:", round(roc_auc_score(d.ev, d.r), 3))
print("is the gap deterministic in #actions? corr(dur, nact) =", round(H[["dur", "nact"]].corr().iloc[0, 1], 3), " residual sd", round(H.dur_res.std(), 3))
