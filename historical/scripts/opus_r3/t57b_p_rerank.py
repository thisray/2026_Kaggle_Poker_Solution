"""R3-P7b: re-rank ONLY inside the top-3000 block (the block keeps its positions), mixing the deployed OOF rank with
the clean substitution LLR rank: order = rank_pct(base|block) + w * rank_pct(llr|block). Official AP, raw and
excluding suspected hidden positives. Also reports each signal's AUC inside the block."""
import numpy as np, pandas as pd, json
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
def ap(y, s):
    o = np.argsort(-s, kind="mergesort"); r = y[o]; tp = np.cumsum(r); k = np.arange(1, len(r) + 1)
    return float(np.sum(tp / k * r) / max(int(y.sum()), 1))
def auc(y, s):
    r = pd.Series(s).rank().values; n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
o = pd.read_parquet(f"{OUT}/m15_v6ens_base_train_oof.parquet"); o = o[o.src == "devsub11"].copy()
lo = o.key // 12000; hi = o.key % 12000; o["slot"] = loc.pool.loc[lo].values * 900 + loc.local.loc[lo].values * 30 + loc.local.loc[hi].values
o["rk"] = o.oof.rank(ascending=False, method="first")
V = pd.read_parquet(f"{OUT}/r3/t41_mid_clean.parquet"); V = V[V.set == "devsub11"][["slot", "llr"]]
o = o.merge(V, on="slot", how="left"); y = o.y.values.astype(int); keep = ~(o.hid.values.astype(bool) & (y == 0))
blk = (o.rk <= 3000).values & o.llr.notna().values
yb = y[blk]; print(f"block {blk.sum()} pairs, positives {int(yb.sum())} | AUC base {auc(yb, -o.rk.values[blk]):.4f} | AUC llr {auc(yb, o.llr.values[blk]):.4f}")
base_score = -o.rk.values.astype(float)                                    # higher = better
rb = pd.Series(-o.rk.values[blk]).rank(pct=True).values; rl = pd.Series(o.llr.values[blk]).rank(pct=True).values
res = {}
for w in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0):
    mix = rb + w * rl
    new = base_score.copy()
    pos = np.sort(base_score[blk])[::-1]                                    # keep the block's own score slots
    new[blk] = pos[np.argsort(np.argsort(-mix, kind="mergesort"), kind="mergesort")]
    res[w] = dict(ap_raw=round(ap(y, new), 5), ap_nohid=round(ap(y[keep], new[keep]), 5), auc_block=round(auc(yb, mix), 4))
    print(f"w={w:.2f} AP_raw {res[w]['ap_raw']:.5f} AP_nohid {res[w]['ap_nohid']:.5f} AUC_block {res[w]['auc_block']:.4f}", flush=True)
json.dump({str(k): v for k, v in res.items()}, open(f"{OUT}/r3/t57b_p_rerank.json", "w"), indent=1)
