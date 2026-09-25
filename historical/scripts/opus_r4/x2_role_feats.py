"""R4-X2a (ORIENT=net default since the Codex review; ORIENT=flow reproduces the first version): role-oriented, label-free hand features for every co-seated hand of a list of pairs (dev or eval).
Orientation: R (receiver) = the member who received more directed chip flow from the partner over ALL co-seated hands of the phase (no labels used).
Usage: python x2_role_feats.py dev|eval <pairs parquet with slot,pa,pb> <out parquet>"""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917")
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
phase = 0 if sys.argv[1] == "dev" else 1; pairs = pd.read_parquet(sys.argv[2])[["slot", "pa", "pb"]].drop_duplicates("slot"); outp = sys.argv[3]
H, S, T, SL = PI.all_pair_hands(phase); need = np.isin(SL, pairs.slot.values); H, S, T, SL = H[need], S[need], T[need], SL[need]
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_act = np.load(f"{D}/a_act.npy"); a_amt = np.load(f"{D}/a_amount.npy"); a_st = np.load(f"{D}/a_st.npy"); a_tc = np.load(f"{D}/a_to_call.npy")
net = np.load(f"{D}/s_net.npy"); con = np.load(f"{D}/s_contrib.npy"); bb = np.load(f"{D}/h_bb.npy").astype(float); sd = np.load(f"{D}/s_sd.npy"); ts = np.load(f"{D}/h_ts.npy"); sp = np.load(f"{D}/s_player.npy")
HS1 = np.load(f"{OUT}/HS1.npy", mmap_mode="r")
df = pd.DataFrame({"slot": SL, "h": H, "s1": S, "s2": T}); df["ts"] = ts[H]
n1 = net[H, S] / bb[H]; n2 = net[H, T] / bb[H]
df["f12"] = np.minimum(np.maximum(-n1, 0), np.maximum(n2, 0)); df["f21"] = np.minimum(np.maximum(-n2, 0), np.maximum(n1, 0))
tot = df.groupby("slot")[["f12", "f21"]].sum()
import os
if os.environ.get("ORIENT", "net") == "flow": recv2 = (tot.f12 >= tot.f21)   # directed partner flow (R4 first version; 93.9% agreement with DT evidence direction on dev)
elif os.environ.get("ORIENT") == "cand":         # directed flow summed over the pair's frozen R15 top-10 candidate hands (label-free; R2-s58: ~100% on dev); fallback: directed flow over all hands
    cd = pd.read_parquet(os.environ["CANDFILE"]).rename(columns={"sl": "slot"}); cd = cd[cd.r <= 10][["slot", "h"]]
    tc = df.merge(cd, on=["slot", "h"]).groupby("slot")[["f12", "f21"]].sum(); recv2 = (tot.f12 >= tot.f21); ok = tc.index[(tc.f12 != tc.f21)]; recv2.loc[ok] = (tc.f12 > tc.f21).loc[ok]
else:                                          # total net of each member over all co-seated hands (the t58 rule; 98.6% agreement on dev)
    df["n1"] = n1; df["n2"] = n2; tn = df.groupby("slot")[["n1", "n2"]].sum(); recv2 = (tn.n2 > tn.n1); df = df.drop(columns=["n1", "n2"])
if os.environ.get("FLIP") == "1": recv2 = ~recv2   # forced opposite orientation (used to score symmetric families with direction-trained models)
df["rs"] = np.where(df.slot.map(recv2), df.s2, df.s1).astype(int); df["ss"] = np.where(df.slot.map(recv2), df.s1, df.s2).astype(int)
df["flow_margin"] = df.slot.map(((tot.f12 - tot.f21).abs() / (tot.f12 + tot.f21 + 1)))
df = df.sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True)
rows = []
for h, r, s in zip(df.h.values, df.rs.values, df.ss.values):
    ks = slice(off[h], off[h + 1]); seat = a_seat[ks]; act = a_act[ks]; st = a_st[ks]; amt = a_amt[ks]; tc = a_tc[ks]; B = bb[h]
    aggr = (act == 3) | (act == 4) | ((act == 5) & (amt > tc)); pre = st == 0; ms = seat == s; mr = seat == r
    fa = np.flatnonzero(aggr & pre); fr = seat[fa[0]] if len(fa) else -1
    s_last = int(act[ms][-1]) if ms.any() else -1; r_last = int(act[mr][-1]) if mr.any() else -1
    s_lst = int(st[ms][-1]) if ms.any() else 0; r_lst = int(st[mr][-1]) if mr.any() else 0
    # last aggressor before S's final action / R's final action
    def last_aggr_before(mask):
        i = np.flatnonzero(mask)
        if not len(i): return -1
        j = np.flatnonzero(aggr[:i[-1]]); return int(seat[j[-1]]) if len(j) else -1
    la_s = last_aggr_before(ms); la_r = last_aggr_before(mr)
    rows.append((net[h, r] / B, net[h, s] / B, con[h, r] / B, con[h, s] / B, 1 * (fr == r) + 2 * (fr == s), float((aggr & pre & ms).any()), float((aggr & ~pre & ms).any()), float((aggr & pre & mr).any()),
                 float((aggr & ~pre & mr).any()), s_last, r_last, s_lst, r_lst, float(((~ms) & (~mr) & ~pre).any()), float(sd[h, s]), float(sd[h, r]), int(st.max()) if len(st) else 0,
                 float(HS1[h, 0, s]), float(HS1[h, 0, r]), float(HS1[h, s_lst, s]), float(HS1[h, r_lst, r]), float(s_last == 0 and la_s == r), float(r_last == 0 and la_r == s),
                 float(((act == 2) & ms).sum()), float(((act <= 2) & ms).sum()), float((aggr & mr).sum()), float((aggr & ms).sum())))
cols = ["netR", "netS", "conR", "conS", "first_raiser", "s_aggr_pre", "s_aggr_post", "r_aggr_pre", "r_aggr_post", "s_last", "r_last", "s_last_st", "r_last_st", "o_post", "sdS", "sdR", "stmax",
        "hsS_pre", "hsR_pre", "hsS_last", "hsR_last", "s_fold_to_r", "r_fold_to_s", "s_calls", "s_passive", "r_aggr_n", "s_aggr_n"]
F = pd.DataFrame(rows, columns=["x_" + c for c in cols]); df = pd.concat([df, F], axis=1)
df["x_dir"] = np.sign(df.x_netR - df.x_netS); df["x_big"] = ((df.x_conR >= 20) & (df.x_conS >= 20)).astype(float); df["x_vol"] = ((df.x_conR > 1) & (df.x_conS > 1)).astype(float)
df["x_bigR"] = df.x_big * (df.x_dir == 1); df["x_cell"] = df.x_vol * (df.x_dir == 1)
g = df.groupby("slot"); df["x_k"] = g.cumcount(); df["x_n"] = g.h.transform("size"); df["x_rel"] = (df.x_k + 0.5) / df.x_n; df["x_inv_n"] = 1.0 / df.x_n
df["x_k_bigR"] = g.x_bigR.cumsum() - df.x_bigR; df["x_k_cell"] = g.x_cell.cumsum() - df.x_cell; df["x_n_cell"] = g.x_cell.transform("sum"); df["x_n_bigR"] = g.x_bigR.transform("sum")
df["x_rel_cell"] = (df.x_k_cell + 0.5) / df.x_n_cell.clip(lower=1); df["x_fold_strength"] = df.x_s_fold_to_r * df.x_hsS_last; df["x_flow_margin"] = df.flow_margin
df.drop(columns=["f12", "f21", "flow_margin", "s1", "s2"]).to_parquet(outp); print("saved", outp, df.shape, "pairs", df.slot.nunique())
