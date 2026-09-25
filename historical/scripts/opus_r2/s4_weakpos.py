"""Autopsy of weak positives in the P model: are their evidence hands detectable at hand level?"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917")
import aggmod as AG
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
T = pd.read_parquet(f"{OUT}/m15_v6base_r2_train_oof.parquet")
d = T[T.src == "devsub11"].copy()
d = d[~d.hid].sort_values("oof", ascending=False).reset_index(drop=True); d["rank"] = np.arange(len(d))
pos = d[d.y == 1]
print("positives", len(pos), "rank quantiles", np.quantile(pos["rank"], [0.5, 0.9, 0.95, 0.98]).round(0))
lr = pd.read_parquet(f"{OUT}/s3_mil_lr_devsub11.parquet")
mil = pd.read_parquet(f"{OUT}/m26_mil_devsub11.parquet")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
def slot_of_key(k):
    a = k // 12000; b = k % 12000
    return loc.pool.loc[a].values * 900 + loc.local.loc[a].values * 30 + loc.local.loc[b].values
pos = pos.assign(slot=slot_of_key(pos.key.values))
pos = pos.merge(mil, on="slot", how="left").merge(lr[["slot", "lr_any_max", "lr_any_rhat", "n_h"]], on="slot", how="left")
pos["weak"] = pos["rank"] > 300
cols = ["n", "mil_cnt999", "mil_cnt99", "mil_top3", "mil_max", "lr_any_max", "lr_any_rhat"]
print(pos.groupby("weak")[cols].median().round(3).to_string())
print(pos.groupby(["weak", "fam"]).size().unstack(0).to_string())
# evidence hands of weak positives: are they inside the devsub11 hand mask, and what are their detector scores?
s0 = np.load(f"{OUT}/m26_handscore_phase0.npy"); sl0 = np.load(f"{OUT}/m26_slot_phase0.npy"); h0 = np.load(f"{OUT}/m26_h_phase0.npy")
hm = AG.hand_mask("devsub", 11)
pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
labels = pd.read_csv(f"{RAW}/development_labels.csv"); ev = pd.read_csv(f"{RAW}/development_evidence.csv")
hidx = pd.read_parquet(f"{OUT}/np/hand_index.parquet"); hmap = dict(zip(hidx.hand_id, hidx.hi))
lo = np.minimum(labels.player_1.map(pmap), labels.player_2.map(pmap)).values; hi = np.maximum(labels.player_1.map(pmap), labels.player_2.map(pmap)).values
labels["slot"] = loc.pool.loc[lo].values * 900 + loc.local.loc[lo].values * 30 + loc.local.loc[hi].values
ev = ev.merge(labels[["pair_id", "slot"]], on="pair_id"); ev["h"] = ev.hand_id.map(hmap)
idx = pd.Series(np.arange(len(sl0)), index=pd.MultiIndex.from_arrays([sl0, h0]))
ev["row"] = idx.reindex(pd.MultiIndex.from_arrays([ev.slot.values, ev.h.values])).values
ev = ev.dropna(subset=["row"]); ev["row"] = ev.row.astype(np.int64)
ev["score"] = s0[ev.row.values]; ev["in_mask"] = hm[ev.h.values] == 1
neg_q = np.quantile(s0[np.random.RandomState(0).choice(len(s0), 2_000_000, replace=False)], [0.99, 0.999])
ev = ev.merge(pos[["slot", "weak", "rank"]], on="slot")
print("neg q99 / q999:", neg_q.round(4))
g = ev.groupby("weak")
print("evidence hands in devsub11 mask frac:", g.in_mask.mean().round(3).to_dict())
print("evidence score > q999 frac:", g.apply(lambda x: (x.score > neg_q[1]).mean()).round(3).to_dict(), " > q99:", g.apply(lambda x: (x.score > neg_q[0]).mean()).round(3).to_dict())
print("in-mask evidence hands per pair:", ev[ev.in_mask].groupby(["weak", "slot"]).size().groupby("weak").mean().round(2).to_dict())
print("in-mask evidence with score>q999 per pair:", ev[ev.in_mask & (ev.score > neg_q[1])].groupby(["weak", "slot"]).size().reindex(pd.MultiIndex.from_frame(pos[["weak", "slot"]]), fill_value=0).groupby("weak").mean().round(2).to_dict())
pos[pos.weak][["slot", "fam", "rank", "oof", "n"] + cols[1:]].to_csv(f"{OUT}/s4_weak_positives.csv", index=False)
