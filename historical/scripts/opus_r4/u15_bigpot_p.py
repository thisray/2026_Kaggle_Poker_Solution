"""R4-U15: does the partner-specific big-pot count add to P in the tail band? Conditional AUC of tail positives vs unlabeled pairs inside OOF rank bands."""
import numpy as np, pandas as pd, sys
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917")
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
H, S, T, SL = PI.all_pair_hands(0)
con = np.load(f"{D}/s_contrib.npy"); net = np.load(f"{D}/s_net.npy"); bb = np.load(f"{D}/h_bb.npy").astype(float); sp = np.load(f"{D}/s_player.npy")
cS = con[H, S] / bb[H]; cT = con[H, T] / bb[H]; nS = net[H, S] / bb[H]; nT = net[H, T] / bb[H]
df = pd.DataFrame({"sl": SL, "big20": (cS >= 20) & (cT >= 20), "big40": (cS >= 40) & (cT >= 40), "flow": np.where((cS >= 20) & (cT >= 20), nS - nT, 0.0), "one": 1})
g = df.groupby("sl").agg(ncs=("one", "sum"), b20=("big20", "sum"), b40=("big40", "sum"), flow=("flow", "sum"))
g["absflow"] = g.flow.abs(); g["r20"] = g.b20 / g.ncs; g["r40"] = g.b40 / g.ncs
# player-level outsider baseline: big-pot rate of each player over all his pair-hands
pl = pd.DataFrame({"p": np.r_[sp[H, S], sp[H, T]], "b": np.r_[df.big20.values, df.big20.values]}).groupby("p").b.mean()
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
o = pd.read_parquet(f"{OUT}/m15_o_pos_a_train_oof.parquet"); x = o[o.src == "devsub11"].copy(); x["p_lo"] = x.key // 12000; x["p_hi"] = x.key % 12000
x["sl"] = x.pool * 900 + loc.local.loc[x.p_lo].values * 30 + loc.local.loc[x.p_hi].values
x = x.merge(g, left_on="sl", right_index=True, how="left"); 
x["z20"] = (x.b20 - x.ncs * 0.0126) / np.sqrt(np.maximum(x.ncs * 0.0126, 0.5)); x["rank"] = x.oof.rank(ascending=False, method="first")
print("overall AUC (labelled pos vs unlabeled)", {c: round(roc_auc_score((x.label == 1)[x.label != 0], x[c][x.label != 0].fillna(0)), 3) for c in ["b20", "z20", "absflow", "r40"]})
for lo, hi in [(200, 600), (300, 1000), (300, 3000), (450, 5000)]:
    band = x[(x["rank"] > lo) & (x["rank"] <= hi) & (x.label != 0)]; y = (band.label == 1).astype(int)
    print(f"band ({lo},{hi}] n={len(band)} pos={int(y.sum())} | AUC oof {roc_auc_score(y, band.oof):.3f} | " + " ".join(f"{c} {roc_auc_score(y, band[c].fillna(0)):.3f}" for c in ["b20", "z20", "absflow", "r40"]))
    for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]:
        bb_ = band[(band.label == -1) | (band.fam == fam)]; yy = (bb_.label == 1).astype(int)
        if yy.sum() >= 3: print(f"    {fam[:2]} pos={int(yy.sum())} z20 AUC {roc_auc_score(yy, bb_.z20.fillna(0)):.3f} absflow {roc_auc_score(yy, bb_.absflow.fillna(0)):.3f}")
