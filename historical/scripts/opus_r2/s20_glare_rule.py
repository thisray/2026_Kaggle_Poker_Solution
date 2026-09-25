"""Among in-window dominant-direction glaring hands (collusion-like), what separates evidence from non-evidence?
Candidate labeler rules: value actually transferred to the partner (flow donor->receiver, receiver net>0), street, pot size, #players."""
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
A = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A}/opus_r1_20260917"; D_ = f"{OUT}/np"; RAW = f"{A}/data/raw"
RN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[0][2:].split(","); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
R = np.load(f"{OUT}/R_v1.npy", mmap_mode="r"); Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); sp = np.load(f"{D_}/s_player.npy")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
G = pd.read_parquet(f"{OUT}/s19_glare.parquet")
M = pd.read_parquet(f"{OUT}/m25e1_handfeat2_m19w10_oof.parquet")[["sl", "h"]]
G = G[G.win & G.g_dom].copy()
sl = G.sl.values; h = G.h.values
plo = mem[sl // 900, (sl % 900) // 30]; phi = mem[sl // 900, sl % 30]
sa = np.argmax(sp[h] == plo[:, None], axis=1); sb = np.argmax(sp[h] == phi[:, None], axis=1)
# donor = the one who folded with high equity in the dominant direction
fab = np.asarray(R[h, sa, sb, 1]); eab = np.asarray(R[h, sa, sb, 15])
gab = (fab > 0) & (eab / np.maximum(fab, 1) >= 0.7)
don = np.where(gab, sa, sb); rec = np.where(gab, sb, sa)
ri = lambda n: RN.index(n); pi = lambda n: PN.index(n)
G["flow_dr"] = np.asarray(R[h, don, rec, ri("flow")]); G["flow_rd"] = np.asarray(R[h, rec, don, ri("flow")])
G["rec_net"] = np.asarray(Pt[h, rec, pi("net_bb")]); G["don_net"] = np.asarray(Pt[h, don, pi("net_bb")])
G["rec_won"] = np.asarray(Pt[h, rec, pi("won")]); G["don_contrib"] = np.asarray(Pt[h, don, pi("contrib_bb")]); G["rec_contrib"] = np.asarray(Pt[h, rec, pi("contrib_bb")])
G["don_fold_street"] = np.asarray(Pt[h, don, pi("fold_street")]); G["rec_sd"] = np.asarray(Pt[h, rec, pi("sd")])
G["eq_fold"] = eab / np.maximum(fab, 1); G.loc[~gab, "eq_fold"] = np.asarray(R[h[~gab], sb[~gab], sa[~gab], 15]) / np.maximum(np.asarray(R[h[~gab], sb[~gab], sa[~gab], 1]), 1)
G["don_pf_eq"] = np.asarray(Pt[h, don, pi("pf_eq_rand")]); G["don_eq_last"] = np.asarray(Pt[h, don, pi("eq_last")])
G["n_folds_to"] = np.where(gab, fab, np.asarray(R[h, sb, sa, 1]))
G["hu_streets"] = np.asarray(R[h, sa, sb, ri("hu_streets")])
for fm, F in G.groupby("fam"):
    print(f"==== {fm}: n {len(F)}  P(ev) {F.ev.mean():.3f}")
    for c, bins in [("rec_won", [-.1, .5, 1.1]), ("flow_dr", [-1, 0, 1e9]), ("rec_net", [-1e9, 0, 1e9]), ("don_fold_street", [-.5, .5, 1.5, 2.5, 3.5]), ("don_contrib", [-1, 1, 3, 8, 1e9]), ("eq_fold", [0.69, .8, .9, 1.01]), ("don_pf_eq", [0, .45, .55, .65, 1]), ("rec_sd", [-.1, .5, 1.1])]:
        t = F.groupby(pd.cut(F[c], bins), observed=True).ev.agg(["mean", "size"]).round(3)
        print(f"   {c:16s}", {str(k): tuple(v) for k, v in t.iterrows()})
    X = F[["rec_won", "flow_dr", "rec_net", "don_net", "don_contrib", "rec_contrib", "don_fold_street", "eq_fold", "don_pf_eq", "don_eq_last", "n_folds_to", "hu_streets", "rec_sd"]].fillna(-1)
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=15).fit(X, F.ev)
    print(export_text(dt, feature_names=list(X.columns), show_weights=True))
