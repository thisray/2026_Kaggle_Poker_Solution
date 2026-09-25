"""Round-14 P v2 diagnostics: standalone AP of new features + audit prep."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

S = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round11_scoped"
pe = pd.read_parquet(f"{S}/r14_pexp_v2/pair_entry_expected_v2.parquet")
rw = pd.read_parquet(f"{S}/r14_pexp_v2/p_expected_rowwise.parquet")
rw["phase"] = np.where(rw.src == "eval_lab", 1, 0)
rw["plo"] = rw.key // 12000
rw["phi"] = rw.key % 12000
m = rw.merge(pe, on=["pool", "phase", "plo", "phi"], how="left")
print("merged", m.shape, "nan pe", int(m.pe_n.isna().sum()))
for src in ["devsub11", "devsub12"]:
    t = m[m.src == src]
    y = (t.y == 1).to_numpy(int)
    for col in ["pe_cond_z", "pe_cond_shrunk", "pe_cond_rate", "pe_z_both", "pe_cond_gap",
                "pe_lbf_d1.0_r0.1", "pe_lbf_d2.0_r0.1"]:
        v = t[col].fillna(0).to_numpy(float)
        ap = average_precision_score(y, v)
        apn = average_precision_score(y, -v)
        print(f"{src} {col}: AP(+)={ap:.4f} AP(-)={max(ap,apn):.4f} sign={'+' if ap>=apn else '-'}")
    print(f"{src} base_score AP={average_precision_score(y, t.base_score.to_numpy()):.4f} new_score AP={average_precision_score(y, t.new_score.to_numpy()):.4f}")

# audit input for v2
out = m[["pool", "key", "label", "base_score", "new_score"]].copy()
out["pair_id"] = "K" + out.key.astype(str)
out = out.rename(columns={"label": "label"})[["pair_id", "pool", "label", "base_score", "new_score"]]
out = out.drop_duplicates("pair_id", keep="first").reset_index(drop=True)
out.to_csv(f"{S}/r14_pexp_v2/p_expected_v2_audit.csv", index=False)
print("audit input", out.shape, out.label.value_counts().to_dict())
