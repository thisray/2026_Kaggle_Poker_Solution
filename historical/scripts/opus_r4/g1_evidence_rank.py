"""R4-G1: what orders the evidence list? evidence_rank is NOT chronological (54.7% agreement). Within-pair rank correlation of evidence_rank with candidate quantities."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; O = f"{A_}/opus_r1_20260917"; RAW = f"{A_}/data/raw"; pd.set_option("display.width", 220)
ev = pd.read_csv(f"{RAW}/development_evidence.csv"); hi = pd.read_parquet(f"{O}/np/hand_index.parquet").set_index("hand_id").hi; ev["h"] = ev.hand_id.map(hi)
lab = pd.read_csv(f"{RAW}/development_labels.csv"); pidx = pd.read_parquet(f"{O}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
lab["pa"] = lab.player_1.map(pmap); lab["pb"] = lab.player_2.map(pmap); ev = ev.merge(lab[["pair_id", "pa", "pb"]], on="pair_id")
D = f"{O}/np"; ts = np.load(f"{D}/h_ts.npy"); pot = np.load(f"{D}/h_pot.npy"); bb = np.load(f"{D}/h_bb.npy").astype(float); sp = np.load(f"{D}/s_player.npy"); net = np.load(f"{D}/s_net.npy"); con = np.load(f"{D}/s_contrib.npy")
off = np.load(f"{D}/a_off.npy"); HS1 = np.load(f"{O}/HS1.npy", mmap_mode="r"); P = np.load(f"{O}/P_v1.npy", mmap_mode="r")
rows = []
for h, pa, pb in zip(ev.h.values, ev.pa.values, ev.pb.values):
    s = sp[h]; a = int(np.flatnonzero(s == pa)[0]); b = int(np.flatnonzero(s == pb)[0]); B = bb[h]
    rows.append((ts[h], h, pot[h] / B, abs(net[h, a] - net[h, b]) / B, (net[h, a] + net[h, b]) / B, max(net[h, a], net[h, b]) / B, min(net[h, a], net[h, b]) / B, (con[h, a] + con[h, b]) / B, off[h + 1] - off[h],
                 float(P[h, a, 12]), float(P[h, b, 12]), max(float(P[h, a, 12]), float(P[h, b, 12])), abs(float(P[h, a, 12]) - float(P[h, b, 12])), float(P[h, a, 16]) + float(P[h, b, 16])))
cols = ["ts", "h", "pot_bb", "abs_net_diff", "pair_net", "max_net", "min_net", "pair_contrib", "n_actions", "eqA", "eqB", "eq_max", "eq_gap", "eq_last_sum"]
for j, c in enumerate(cols): ev[c] = [r[j] for r in rows]
ev["chron"] = ev.groupby("pair_id").ts.rank()
res = {}
for c in cols + ["chron"]:
    v = [spearmanr(g.evidence_rank, g[c]).correlation for _, g in ev.groupby("pair_id") if len(g) >= 4 and g[c].nunique() > 1]; res[c] = (np.nanmean(v), np.nanmedian(v), len(v))
print(pd.DataFrame(res, index=["mean within-pair spearman", "median", "pairs"]).T.round(3).sort_values("mean within-pair spearman").to_string())
for fam, g in ev.groupby("behavior_family"):
    print(fam, "rank==chron", (g.evidence_rank == g.chron).mean().round(3), "| mean pot by evidence_rank:", g.groupby("evidence_rank").pot_bb.median().round(1).tolist(), "| mean chron by rank:", g.groupby("evidence_rank").chron.mean().round(2).tolist())
# is the permutation pattern structured? distribution of the chronological position of rank-1
print(ev[ev.evidence_rank == 1].groupby("behavior_family").chron.value_counts().unstack().to_string())
ev.to_parquet(f"{O}/r4/g1_evidence_rank.parquet")
