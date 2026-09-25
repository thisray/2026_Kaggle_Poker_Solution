"""DT: evidence vs same-direction high-score non-evidence hands before the last evidence (within the same pairs)."""
import numpy as np, pandas as pd, json
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D_ = f"{OUT}/np"
M = pd.read_parquet(f"{OUT}/m6_handscores.parquet"); q999 = json.load(open(f"{OUT}/m8_evrank_meta.json"))["q999"]
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); members = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): members[pool, g.local.values] = g.player_gi.values
sp = np.load(f"{D_}/s_player.npy"); net = np.load(f"{D_}/s_net.npy"); bb = np.load(f"{D_}/h_bb.npy"); contrib = np.load(f"{D_}/s_contrib.npy"); pot = np.load(f"{D_}/h_pot.npy")
sd = np.load(f"{D_}/s_sd.npy"); folded = np.load(f"{D_}/s_folded.npy"); nsd = np.load(f"{D_}/h_nsd.npy"); board = np.load(f"{D_}/h_board.npy")
R = np.load(f"{OUT}/R_v1.npy", mmap_mode="r"); P1 = np.load(f"{OUT}/P_v1.npy", mmap_mode="r")
F = M[(M.pos) & (M.fam == "directed_transfer")].copy()
sl = F.sl.values; h = F.h.values
plo = members[sl // 900, (sl % 900) // 30]; phi = members[sl // 900, sl % 30]
sa = np.argmax(sp[h] == plo[:, None], axis=1); sb = np.argmax(sp[h] == phi[:, None], axis=1)
F["flowAB"] = np.asarray(R[h, sa, sb, 7]); F["flowBA"] = np.asarray(R[h, sb, sa, 7])
# donor per pair from evidence
ev = F[F.ev]; dsign = (ev.flowAB - ev.flowBA).groupby(ev.sl).sum()
F["donorA"] = F.sl.map(dsign) > 0
d_seat = np.where(F.donorA, sa, sb); r_seat = np.where(F.donorA, sb, sa)
F["flow_dr"] = np.asarray(R[h, d_seat, r_seat, 7]); F["flow_rd"] = np.asarray(R[h, r_seat, d_seat, 7])
F["d_net"] = net[h, d_seat] / bb[h]; F["r_net"] = net[h, r_seat] / bb[h]
F["d_contrib"] = contrib[h, d_seat] / bb[h]; F["pot_bb"] = pot[h] / bb[h]
F["d_folded"] = folded[h, d_seat]; F["d_sd"] = sd[h, d_seat]; F["r_sd"] = sd[h, r_seat]; F["nsd"] = nsd[h]
F["nboard"] = (board[h] >= 0).sum(1)
F["d_fold_to_r"] = np.asarray(R[h, d_seat, r_seat, 1]); F["r_fold_to_d"] = np.asarray(R[h, r_seat, d_seat, 1])
F["d_call_to_r"] = np.asarray(R[h, d_seat, r_seat, 2]); F["d_raise_over_r"] = np.asarray(R[h, d_seat, r_seat, 3]); F["r_raise_over_d"] = np.asarray(R[h, r_seat, d_seat, 3])
F["d_eq_fold_to_r"] = np.asarray(R[h, d_seat, r_seat, 15]); F["d_eq_call_to_r"] = np.asarray(R[h, d_seat, r_seat, 16])
F["d_sunk_fold"] = np.asarray(R[h, d_seat, r_seat, 5]); F["hu_streets"] = np.asarray(R[h, d_seat, r_seat, 13])
F["d_pf"] = np.asarray(P1[h, d_seat, 12]); F["r_pf"] = np.asarray(P1[h, r_seat, 12]); F["d_eq_last"] = np.asarray(P1[h, d_seat, 16]); F["r_eq_last"] = np.asarray(P1[h, r_seat, 16])
F["d_lost_to_r"] = np.minimum(np.maximum(-F.d_net, 0), np.maximum(F.r_net, 0))
last_ev = F[F.ev].groupby("sl").ts.max()
F["before_last"] = F.ts < F.sl.map(last_ev)
grpE = F[F.ev]; grpX = F[(~F.ev) & F.before_last & (F.s > q999) & (F.flow_dr > F.flow_rd)]; grpN = F[(~F.ev) & (F.s <= 0.05)]
cols = ["flow_dr", "flow_rd", "d_lost_to_r", "d_net", "r_net", "d_contrib", "pot_bb", "d_folded", "d_sd", "r_sd", "nsd", "nboard", "d_fold_to_r", "r_fold_to_d", "d_call_to_r", "d_raise_over_r", "r_raise_over_d",
        "d_eq_fold_to_r", "d_eq_call_to_r", "d_sunk_fold", "hu_streets", "d_pf", "r_pf", "d_eq_last", "r_eq_last"]
out = pd.DataFrame({"EVIDENCE": grpE[cols].mean(), "X_same_dir_nonev": grpX[cols].mean(), "normal_hands": grpN[cols].mean()}).round(3)
print("counts: evidence", len(grpE), " X same-dir nonev", len(grpX), " normal", len(grpN)); print(out.to_string())
for c in ["d_lost_to_r", "flow_dr", "pot_bb", "d_contrib"]:
    print(c, "evidence quantiles", np.round(np.quantile(grpE[c], [0.1, 0.25, 0.5, 0.75]), 2), " X quantiles", np.round(np.quantile(grpX[c], [0.1, 0.25, 0.5, 0.75]), 2))
print("evidence: fraction where donor folded:", round(grpE.d_folded.mean(), 3), " donor at showdown:", round(grpE.d_sd.mean(), 3), "; X:", round(grpX.d_folded.mean(), 3), round(grpX.d_sd.mean(), 3))
# street of donor's last action/fold
F.to_parquet(f"{OUT}/a13_dt_hands.parquet")
