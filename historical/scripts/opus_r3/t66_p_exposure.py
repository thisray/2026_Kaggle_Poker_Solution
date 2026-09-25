"""R3-P8: where does the pair-AP loss sit by exposure (shared hands) and family, and is the n>=38 training filter
leaving an unmodelled regime? Dev OOF of the deployed P ensemble, official AP with the hidden positives dropped."""
import numpy as np, pandas as pd, json
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
def ap(y, s):
    o = np.argsort(-s, kind="mergesort"); r = y[o]; tp = np.cumsum(r); k = np.arange(1, len(r) + 1)
    return float(np.sum(tp / k * r) / max(int(y.sum()), 1))
o = pd.read_parquet(f"{OUT}/m15_v6ens_base_train_oof.parquet")
for src in ("devsub11", "devsub12"):
    d = o[(o.src == src)].copy(); d = d[~d.hid.astype(bool) | (d.y == 1)]
    d["rk"] = d.oof.rank(ascending=False, method="first")
    base = ap(d.y.values, d.oof.values)
    print(f"== {src}: pairs {len(d)}, positives {int(d.y.sum())}, AP {base:.5f}")
    print("   exposure (n) distribution of positives:", d[d.y == 1].n.describe(percentiles=[.05, .1, .25, .5]).round(1)[["min", "5%", "10%", "25%", "50%", "max"]].to_dict())
    for lo, hi in ((0, 38), (38, 60), (60, 90), (90, 10000)):
        m = (d.n >= lo) & (d.n < hi); pos = d[m & (d.y == 1)]
        if not len(pos): continue
        # AP contribution lost by these positives (precision at their rank vs 1.0)
        prec = [(d[(d.rk <= r)].y.sum()) / r for r in pos.rk.values]
        print(f"   n in [{lo},{hi}): pairs {int(m.sum())}, positives {len(pos)}, median rank {pos.rk.median():.0f}, mean precision {np.mean(prec):.3f}, AP loss {(len(pos) - np.sum(prec)) / int(d.y.sum()):.5f}")
    for fam in ("directed_transfer", "soft_play", "coordinated_isolation"):
        pos = d[(d.y == 1) & (d.fam == fam)]
        prec = [(d[(d.rk <= r)].y.sum()) / r for r in pos.rk.values]
        print(f"   {fam}: positives {len(pos)}, median rank {pos.rk.median():.0f}, AP loss {(len(pos) - np.sum(prec)) / int(d.y.sum()):.5f}")
ev = pd.read_parquet(f"{OUT}/m15_v6ens_base_eval_scores.parquet") if __import__("os").path.exists(f"{OUT}/m15_v6ens_base_eval_scores.parquet") else None
if ev is not None:
    print("eval pairs scored:", len(ev), "| n<38:", int((ev.n < 38).sum()), "| n in [20,38):", int(((ev.n >= 20) & (ev.n < 38)).sum()))
    ev["rk"] = ev.score.rank(ascending=False, method="first")
    print("   of the eval top-600 by this model, n<38:", int((ev[ev.rk <= 600].n < 38).sum()))
