"""Outcome conditions of labelled evidence hands (known families, dev): do evidence hands require a realised value
movement (pair net > 0, partner wins, flow between members) that non-evidence candidate hands lack?"""
import numpy as np, pandas as pd
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D_ = f"{OUT}/np"
L = open(f"{OUT}/feature_names_v1.txt").read().split("\n"); RN = L[0][2:].split(",")
R = np.load(f"{OUT}/R_v1.npy", mmap_mode="r"); sp = np.load(f"{D_}/s_player.npy", mmap_mode="r")
snet = np.load(f"{D_}/s_net.npy", mmap_mode="r"); swon = np.load(f"{D_}/s_won.npy", mmap_mode="r")
ssd = np.load(f"{D_}/s_sd.npy", mmap_mode="r"); sfold = np.load(f"{D_}/s_folded.npy", mmap_mode="r"); bb = np.load(f"{D_}/h_bb.npy", mmap_mode="r")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
M = pd.read_parquet(f"{OUT}/m25e1_handfeat2_m19w10_oof.parquet")[["sl", "h", "ev", "fam", "ts"]].sort_values(["sl", "ts"]).reset_index(drop=True)
H = M.h.values
plo = mem[M.sl.values // 900, (M.sl.values % 900) // 30]; phi = mem[M.sl.values // 900, M.sl.values % 30]
spH = np.asarray(sp[H]); sa = np.argmax(spH == plo[:, None], axis=1); sb = np.argmax(spH == phi[:, None], axis=1)
ix = np.arange(len(M)); b_ = np.asarray(bb[H]).astype(float)
netH = np.asarray(snet[H]).astype(float); M["na"] = netH[ix, sa] / b_; M["nb"] = netH[ix, sb] / b_
M["pair_net"] = M.na + M.nb
wonH = np.asarray(swon[H]); M["wa"] = wonH[ix, sa] > 0; M["wb"] = wonH[ix, sb] > 0
sdH = np.asarray(ssd[H]); M["sd_any"] = sdH.sum(1) > 0
fH = np.asarray(sfold[H]); M["fa"] = fH[ix, sa] > 0; M["fb"] = fH[ix, sb] > 0
g = lambda i, j, c: np.asarray(R[H, i, j, RN.index(c)]).astype(float)
M["flow_ab"] = g(sa, sb, "flow"); M["flow_ba"] = g(sb, sa, "flow")
M["iso"] = g(sa, sb, "iso_ofold") + g(sb, sa, "iso_ofold")
M["both_in"] = ((g(sa, sb, "opp_active") + g(sb, sa, "opp_active")) > 0)
last = M[M.ev].groupby("sl").ts.max(); M["win_end"] = M.sl.map(last); M["zone"] = np.where(M.ev, "ev", np.where(M.ts <= M.win_end, "in_non", "post"))
M["pair_win"] = M.wa | M.wb; M["pos_net"] = M.pair_net > 0; M["xfer"] = (M.flow_ab != 0) | (M.flow_ba != 0)
M["one_member_net_pos"] = (M.na > 0) | (M.nb > 0)
cols = ["pos_net", "pair_win", "one_member_net_pos", "xfer", "sd_any", "both_in", "fa", "fb"]
pd.set_option("display.width", 200)
for fam, F in M.groupby("fam"):
    print(f"== {fam}")
    T = F.groupby("zone")[cols].mean().round(3); T["n"] = F.groupby("zone").size(); print(T)
    T2 = F.groupby("zone")[["pair_net", "na", "nb"]].median().round(2); print(T2)
    # P(ev | condition) inside the window (ev + in_non)
    W = F[F.zone != "post"]
    for c in ["pos_net", "pair_win", "xfer", "both_in"]:
        print(f"   P(ev | {c}=1) = {W[W[c]].ev.mean():.3f} (n={W[c].sum()}),  P(ev | {c}=0) = {W[~W[c]].ev.mean():.3f} (n={(~W[c]).sum()})")
M.to_parquet(f"{OUT}/s66_dev_outcomes.parquet")
