"""R3-E11: does policy memorisation hurt known-family evidence retrieval?
policy_v1 (source of dec_surprisal_v1 and of R15 surprisal features) trained on RandomState(0).choice(N, 4M) decisions.
For every dev candidate hand of a labelled positive pair (dev OOF, R15 blend as in t3), compute the share of the two
members' decisions that v1 trained on; compare the hit rate of TRUE evidence hands (in the top 5 or not) and the pick rate
of non-evidence hands across exposure bins, per family."""
import numpy as np, pandas as pd
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; RAW = f"{A_}/data/raw"; D = f"{OUT}/np"
d = pd.read_parquet(f"{A_}/round11_scoped/dev_oof_aligned.parquet")[["slot", "hand_id", "ev", "rs_blend"]]
tab = pd.read_csv(f"{A_}/round15_campaign/tabicl_cv/predictions.csv.gz")[["slot", "hand_id", "score"]].rename(columns={"score": "tab"})
d = d.merge(tab, on=["slot", "hand_id"]); d["b"] = 0.6 * d.groupby("slot").tab.rank(pct=True) + 0.4 * d.groupby("slot").rs_blend.rank(pct=True)
d["rank"] = d.groupby("slot").b.rank(ascending=False, method="first"); d["top5"] = d["rank"] <= 5
hidx = pd.read_parquet(f"{D}/hand_index.parquet").set_index("hand_id").hi
pidx = pd.read_parquet(f"{D}/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
lab = pd.read_csv(f"{RAW}/development_labels.csv"); lab = lab[lab.label == 1].copy()
a = lab.player_1.map(pmap).values; b_ = lab.player_2.map(pmap).values; lo = np.minimum(a, b_); hi = np.maximum(a, b_)
lab["slot"] = loc.pool.loc[lo].values * 900 + loc.local.loc[lo].values * 30 + loc.local.loc[hi].values; lab["pa"] = a; lab["pb"] = b_
d = d.merge(lab[["slot", "pa", "pb", "behavior_family"]], on="slot")
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy", mmap_mode="r"); sp = np.load(f"{D}/s_player.npy", mmap_mode="r")
N = int(off[-1]); in1 = np.zeros(N, bool); in1[np.random.RandomState(0).choice(N, 4_000_000, replace=False)] = True
in2 = np.zeros(N, bool); in2[np.random.RandomState(1).choice(N, 6_000_000, replace=False)] = True
hh = hidx.loc[d.hand_id].values; sh1 = np.zeros(len(d)); sh2 = np.zeros(len(d)); nd = np.zeros(len(d), int)
for i, (h, pa_, pb_) in enumerate(zip(hh, d.pa.values, d.pb.values)):
    ks = np.arange(off[h], off[h + 1]); pl = np.asarray(sp[h])[np.asarray(a_seat[ks])]; m = (pl == pa_) | (pl == pb_); ks = ks[m]
    nd[i] = len(ks)
    if len(ks): sh1[i] = in1[ks].mean(); sh2[i] = in2[ks].mean()
d["sh1"] = sh1; d["sh2"] = sh2; d["nd"] = nd
d["bin1"] = pd.cut(d.sh1, [-0.01, 0.0, 0.34, 0.67, 1.0], labels=["0", "(0,1/3]", "(1/3,2/3]", ">2/3"])
pd.set_option("display.width", 200)
print(f"dev candidate hands {len(d)}, positive pairs {d.slot.nunique()}, true evidence hands {int(d.ev.sum())}; mean v1-train share {d.sh1.mean():.3f}")
t = d[d.ev == 1].groupby(["behavior_family", "bin1"], observed=True).agg(n=("top5", "size"), hit=("top5", "mean"))
print("TRUE evidence hands: hit rate (in top 5) by share of member decisions v1 trained on"); print(t.round(3).to_string())
f = d[d.ev == 0].groupby(["behavior_family", "bin1"], observed=True).agg(n=("top5", "size"), picked=("top5", "mean"))
print("NON-evidence candidates: pick rate by the same exposure"); print(f.round(3).to_string())
d[["slot", "hand_id", "ev", "top5", "sh1", "sh2", "nd", "behavior_family"]].to_parquet(f"{OUT}/r3/t44_known_e_memo.parquet")
