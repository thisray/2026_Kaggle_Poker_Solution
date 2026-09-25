"""Local official-semantics decomposition on the exposure-matched dev population (hidden positives excluded)."""
import numpy as np, pandas as pd, lightgbm as lgb, sys
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker/src")
from pokerlab.metrics import stable_ap
from pairfeat import build
from pairfeat2 import build2
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
SRC = sys.argv[1] if len(sys.argv) > 1 else "m5_both_train_oof.parquet"
FAMS = ["directed_transfer", "soft_play", "coordinated_isolation"]
T = pd.read_parquet(f"{OUT}/{SRC}"); ref = pd.read_parquet(f"{OUT}/m5_both_train_oof.parquet")
d = T[T.src == "devsub11"].copy()
hid = set(ref[(ref.src == "devsub11") & (ref.label == -1) & (ref.oof > 0.3)].key)
d = d[~d.key.isin(hid)].sort_values("key", kind="mergesort").reset_index(drop=True)
t = pd.read_parquet(f"{OUT}/ptab_devsub11.parquet"); t["key"] = t.p_lo * 12000 + t.p_hi; t = t.set_index("key").loc[d.key].reset_index()
X = pd.concat([build(t), build2(t)], axis=1)
pos = d.y.values == 1; fam = d.fam.values
yf = pd.Series(fam).map({f: i for i, f in enumerate(FAMS)}).values
pred_fam = np.empty(len(d), dtype=object)
for f in range(5):
    tr = pos & (d.fold.values != f); va = d.fold.values == f
    m = lgb.train(dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=15, min_data_in_leaf=10, feature_fraction=0.6, verbose=-1, num_threads=4), lgb.Dataset(X[tr], yf[tr].astype(int)), 300)
    pred_fam[va] = np.array(FAMS)[m.predict(X[va]).argmax(1)]
risk = d.oof.values; y = d.y.values
P = stable_ap(y, risk)
fam_ap = {f: stable_ap((fam == f).astype(int), risk * (pred_fam == f)) for f in FAMS}
B = float(np.mean(list(fam_ap.values())))
print(f"{SRC}: population {len(d)} positives {int(y.sum())}  P={P:.4f}  B={B:.4f}  family APs", {k: round(v, 4) for k, v in fam_ap.items()})
for E in [0.40, 0.50, 0.60, 0.80]:
    print(f"   S if E={E:.2f}: {0.7 * P + 0.2 * E + 0.1 * B:.4f}")
