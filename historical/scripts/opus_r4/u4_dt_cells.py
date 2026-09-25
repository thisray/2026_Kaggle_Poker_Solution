"""R4-U4: DT (and SP/CI) hand descriptors by role (receiver R / sender S from evidence flow); evidence rate per cell, split by pot size."""
import numpy as np, pandas as pd, sys
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
fam = sys.argv[1]
d = pd.read_parquet(f"{OUT}/t5_dev_seq.parquet").sort_values(["sl", "ts", "h"]).reset_index(drop=True)
d = d[d.fam == fam].copy()
t58 = pd.read_parquet(f"{OUT}/r3/t58_seq_feats.parquet")[["slot", "h", "pa", "pb"]].rename(columns={"slot": "sl"})
d = d.merge(t58, on=["sl", "h"])
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy", mmap_mode="r"); sp = np.load(f"{D}/s_player.npy", mmap_mode="r")
a_act = np.load(f"{D}/a_act.npy", mmap_mode="r"); a_amt = np.load(f"{D}/a_amount.npy", mmap_mode="r"); a_st = np.load(f"{D}/a_st.npy", mmap_mode="r"); a_tc = np.load(f"{D}/a_to_call.npy", mmap_mode="r")
net = np.load(f"{D}/s_net.npy", mmap_mode="r"); con = np.load(f"{D}/s_contrib.npy", mmap_mode="r"); bb = np.load(f"{D}/h_bb.npy", mmap_mode="r"); sd = np.load(f"{D}/s_sd.npy", mmap_mode="r"); fol = np.load(f"{D}/s_folded.npy", mmap_mode="r")
rows = []
for sl, h, pa, pb in zip(d.sl.values, d.h.values, d.pa.values, d.pb.values):
    seats = np.asarray(sp[h]); sa = int(np.flatnonzero(seats == pa)[0]); sb = int(np.flatnonzero(seats == pb)[0])
    rows.append((sa, sb, net[h, sa] / bb[h], net[h, sb] / bb[h]))
d[["sa", "sb", "netA", "netB"]] = np.array(rows)
evn = d[d.ev].groupby("sl")[["netA", "netB"]].sum(); recvA = (evn.netA > evn.netB).to_dict(); d["recvA"] = d.sl.map(recvA)
out = []
for h, sa, sb, ra in zip(d.h.values, d.sa.values.astype(int), d.sb.values.astype(int), d.recvA.values):
    r, s = (sa, sb) if ra else (sb, sa)
    ks = np.arange(off[h], off[h + 1]); seat = np.asarray(a_seat[ks]); act = np.asarray(a_act[ks]); st = np.asarray(a_st[ks]); amt = np.asarray(a_amt[ks]); tc = np.asarray(a_tc[ks])
    aggr = (act == 3) | (act == 4) | ((act == 5) & (amt > tc))
    pre = st == 0
    # first voluntary preflop aggressor
    fa = np.flatnonzero(aggr & pre); first_raiser = seat[fa[0]] if len(fa) else -1
    s_aggr_pre = bool((aggr & pre & (seat == s)).any()); s_aggr_post = bool((aggr & ~pre & (seat == s)).any())
    r_aggr_pre = bool((aggr & pre & (seat == r)).any()); r_aggr_post = bool((aggr & ~pre & (seat == r)).any())
    s_last = act[seat == s][-1] if (seat == s).any() else -1; r_last = act[seat == r][-1] if (seat == r).any() else -1
    o_post = bool(((seat != s) & (seat != r) & ~pre).any())
    out.append((net[h, r] / bb[h], net[h, s] / bb[h], con[h, r] / bb[h], con[h, s] / bb[h], int(first_raiser == r) + 2 * int(first_raiser == s), s_aggr_pre, s_aggr_post, r_aggr_pre, r_aggr_post,
                int(s_last), int(r_last), o_post, bool(sd[h, s]), bool(sd[h, r]), int(st.max())))
cols = ["netR", "netS", "conR", "conS", "first_raiser", "s_aggr_pre", "s_aggr_post", "r_aggr_pre", "r_aggr_post", "s_last", "r_last", "o_post", "sdS", "sdR", "stmax"]
for j, c in enumerate(cols): d[c] = [o[j] for o in out]
d.to_parquet(f"{OUT}/r4/u4_{fam}.parquet")
w = d[d.zone != "post"].copy(); w["big"] = (w.conR >= 20) & (w.conS >= 20); w["both_vol"] = (w.conR > 1) & (w.conS > 1)
w["dir"] = np.sign(w.netR - w.netS).astype(int)
print("evidence: dir", w[w.ev].dir.value_counts().to_dict(), " s_aggr_pre", w[w.ev].s_aggr_pre.mean(), " s_aggr_post", w[w.ev].s_aggr_post.mean(), " r_aggr_any", (w[w.ev].r_aggr_pre | w[w.ev].r_aggr_post).mean())
for nm, x in [("BIG", w[w.big]), ("SMALL both_vol", w[~w.big & w.both_vol])]:
    print("=====", nm, len(x), "ev rate", x.ev.mean())
    for c in ["dir", "first_raiser", "s_aggr_pre", "s_aggr_post", "r_aggr_pre", "r_aggr_post", "s_last", "r_last", "o_post", "sdS", "sdR", "stmax"]:
        print(x.groupby(c).ev.agg(["mean", "sum", "size"]).round(3).T.to_string())
    print(x.groupby(["dir", "s_aggr_pre", "s_aggr_post"]).ev.agg(["mean", "sum", "size"]).round(3).to_string())
