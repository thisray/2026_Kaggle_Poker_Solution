"""Prepare audit_pair_rank input from the P expected-count rowwise OOF."""
import numpy as np
import pandas as pd

RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
S = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round11_scoped"
OP = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"

lab = pd.read_csv(f"{RAW}/development_labels.csv")
pidx = pd.read_parquet(f"{OP}/np/player_index.parquet")
pmap = dict(zip(pidx.player_id, pidx.pi))
lo = np.minimum(lab.player_1.map(pmap), lab.player_2.map(pmap))
hi = np.maximum(lab.player_1.map(pmap), lab.player_2.map(pmap))
lab["key"] = lo * 12000 + hi
lab["label"] = lab.label.astype(int)
lab["pair_id_use"] = lab.pair_id
lab = lab.drop_duplicates("key", keep="first")

d = pd.read_parquet(f"{S}/r12_pexp/p_expected_rowwise.parquet")
d = d.merge(lab[["key", "label", "pair_id_use"]], on="key", how="left", suffixes=("", "_lab"))
d["label_use"] = d["label_lab"].fillna(-1).astype(int)
d["pair_id"] = d.pair_id_use.fillna("U" + d.key.astype(str))
out = d[["pair_id", "pool", "label_use", "base_score", "new_score"]].rename(columns={"label_use": "label"})
out = out.drop_duplicates("pair_id", keep="first").reset_index(drop=True)
assert out.label.isin([-1, 0, 1]).all()
assert np.isfinite(out[["base_score", "new_score"]].to_numpy()).all()
out.to_csv(f"{S}/r12_pexp/p_expected_rowwise_audit.csv", index=False)
print("audit input", out.shape, out.label.value_counts().to_dict())
