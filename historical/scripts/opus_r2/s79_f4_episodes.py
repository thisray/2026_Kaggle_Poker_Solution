"""Is fourth-family activity episodic?  Within member pairs, lag autocorrelation of the per-hand active posterior (P1) and
of 'clear informed' events over co-seated hand order, vs within-pair shuffles; and activity by time decile."""
import numpy as np, pandas as pd
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"
Mx = pd.read_parquet(f"{OUT}/s78_f4_pair_alpha.parquet").sort_values(["slot", "ts"]).reset_index(drop=True)
Mx["clear"] = ((Mx.m1 == 1) | (Mx.m2 == 1)).astype(float)
Mx["dev"] = Mx.P1 * Mx.D1
def lagcorr(df, col, lag):
    num = 0.0; den = 0.0; n = 0
    xs = []; ys = []
    for sl, g in df.groupby("slot"):
        v = g[col].values - g[col].values.mean()
        if len(v) > lag: xs.append(v[:-lag]); ys.append(v[lag:])
    x = np.concatenate(xs); y = np.concatenate(ys); return float((x * y).mean() / np.sqrt((x * x).mean() * (y * y).mean()))
rng = np.random.default_rng(0)
for col in ["P1", "dev", "clear"]:
    obs = [round(lagcorr(Mx, col, L), 4) for L in [1, 2, 3, 5, 10]]
    sh = []
    for s in range(20):
        Z = Mx.copy(); Z[col] = Z.groupby("slot")[col].transform(lambda x: rng.permutation(x.values)); sh.append([lagcorr(Z, col, L) for L in [1, 2, 3, 5, 10]])
    sh = np.array(sh)
    print(f"{col:6s} lag-corr obs {obs} | shuffled mean {np.round(sh.mean(0), 4).tolist()} sd {np.round(sh.std(0), 4).tolist()}")
Mx["dec"] = Mx.groupby("slot").k.transform(lambda k: np.floor(10 * k / (k.max() + 1)))
print("activity by within-pair time decile (mean P1 / mean clear):")
print(Mx.groupby("dec")[["P1", "dev", "clear"]].mean().round(3).T.to_string())
