"""R3-P7: can the de-memorised substitution statistic improve the PAIR ranking (P, weight 0.70)?
Dev protocol: devsub11 OOF scores of the deployed P ensemble + dev labels; the clean M3 pair LLR from t41 for the
top-3000 OOF pairs. Score = rank_pct(base) + w * rank_pct(LLR) inside the top-3000 (others keep base). Official
average precision, reported both raw and excluding the suspected hidden positives (the `hid` flag), with the LLR
feature's own AUC for context."""
import numpy as np, pandas as pd, json
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
def ap(y, s):
    o = np.argsort(-s, kind="mergesort"); r = y[o]; tp = np.cumsum(r); k = np.arange(1, len(r) + 1)
    return float(np.sum(tp / k * r) / max(int(y.sum()), 1))
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
o = pd.read_parquet(f"{OUT}/m15_v6ens_base_train_oof.parquet"); o = o[o.src == "devsub11"].copy()
lo = o.key // 12000; hi = o.key % 12000; o["slot"] = loc.pool.loc[lo].values * 900 + loc.local.loc[lo].values * 30 + loc.local.loc[hi].values
o["rk"] = o.oof.rank(ascending=False, method="first")
M = pd.read_parquet(f"{OUT}/r3/t41_mid_clean.parquet"); V = M[M.set == "devsub11"][["slot", "llr", "ndec"]]
o = o.merge(V, on="slot", how="left")
print(f"devsub11 pairs {len(o)}, positives {int(o.y.sum())}, hidden-flag {int(o.hid.sum())}, with clean LLR {int(o.llr.notna().sum())}")
top = o.rk <= 3000
sub = o[top]
print(f"top-3000: positives {int(sub.y.sum())}, LLR coverage {int(sub.llr.notna().sum())}, AUC(llr | top3000) "
      f"{float((pd.Series(sub.llr.fillna(sub.llr.min())).rank().values[sub.y.values == 1].mean() - (sub.y.sum() + 1) / 2) / (len(sub) - sub.y.sum())):.4f}")
base_pct = o.oof.rank(pct=True).values
llr_pct = np.zeros(len(o)); m = top.values & o.llr.notna().values
llr_pct[m] = pd.Series(o.llr[m]).rank(pct=True).values * 0.0 + (o.llr[m].rank(pct=True).values)
y = o.y.values.astype(int); keep = ~(o.hid.values.astype(bool) & (y == 0))     # AP_nohid: drop suspected hidden positives from the negatives
res = {}
for w in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, -0.005, -0.02):
    s = base_pct + w * llr_pct * m
    res[w] = dict(ap_raw=round(ap(y, s), 5), ap_nohid=round(ap(y[keep], s[keep]), 5))
    print(f"w={w:+.3f} AP_raw {res[w]['ap_raw']:.5f} AP_nohid {res[w]['ap_nohid']:.5f}", flush=True)
json.dump({str(k): v for k, v in res.items()}, open(f"{OUT}/r3/t57_p_rerank.json", "w"), indent=1)
