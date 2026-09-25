"""R3-P15: an MLP pair model, for a learner family the fusion does not yet contain.
Same data path, folds, PU handling and output files as t69_pair_variant.py."""
import numpy as np, pandas as pd, time, sys, os, json
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import average_precision_score
from pathlib import Path
src = Path(__file__).with_name("r25_pair_variant.py").read_text()
head = src.split("params = dict(")[0]
PU = sys.argv[1]; MIL = sys.argv[2]; TAG = sys.argv[3]
g = {"__name__": "__main__"}; sys.argv = ["x", PU, MIL, TAG]
exec(compile(head, "head", "exec"), g)
T, XT, X, y, w, tabs, evalp = g["T"], g["XT"], g["X"], g["y"], g["w"], g["tabs"], g["evalp"]
OUT = g["OUT"]
qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200000, random_state=int(os.environ.get("SEED", 7)))
Xall = qt.fit_transform(np.nan_to_num(XT.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
ev = tabs["eval"]; inev = ev.key.isin(set(evalp.key)).values
Xev = qt.transform(np.nan_to_num(X["eval"][inev].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
HID = tuple(int(x) for x in os.environ.get("HIDDEN", "128,64").split(","))
mk = lambda sd: MLPClassifier(hidden_layer_sizes=HID, alpha=float(os.environ.get("ALPHA", 1e-3)), learning_rate_init=float(os.environ.get("LRI", 1e-3)),
                              batch_size=1024, max_iter=int(os.environ.get("EPOCHS", 40)), early_stopping=True, n_iter_no_change=5, random_state=sd)
oof = np.zeros(len(T)); t0 = time.time()
for f in range(5):
    tr = (T.fold.values != f) & (w > 0); va = T.fold.values == f
    m = mk(int(os.environ.get("SEED", 7)) + f); m.fit(Xall[tr], y[tr])
    oof[va] = m.predict_proba(Xall[va])[:, 1]
    print(f"[{time.time()-t0:6.0f}s] fold {f} done", flush=True)
for nm in ["devsub11", "devsub12"]:
    mm = (T.src == nm).values; mc = mm & ~T.hid.values
    print(f"{TAG} {nm}: AP_raw {average_precision_score(T.y.values[mm], oof[mm]):.4f}  AP_clean {average_precision_score(T.y.values[mc], oof[mc]):.4f}", flush=True)
mfull = mk(int(os.environ.get("SEED", 7))); mfull.fit(Xall[w > 0], y[w > 0])
s_eval = mfull.predict_proba(Xev)[:, 1]
ev[inev][["key", "pool", "p_lo", "p_hi", "n"]].assign(score=s_eval).to_parquet(f"{OUT}/m15_{TAG}_eval_scores.parquet")
T[["key", "pool", "fold", "src", "y", "fam", "label", "n", "hid"]].assign(oof=oof).to_parquet(f"{OUT}/m15_{TAG}_train_oof.parquet")
print("saved", TAG)
