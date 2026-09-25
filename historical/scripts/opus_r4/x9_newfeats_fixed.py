"""R4-X9: corrected re-implementation of the t52/t58 role/strength/surprisal block.
The original (opus_r3/t58, t61) computes nets as s_won - contributions, but s_won is the WON SHARE (0..1), not chips, so its nets are ~ -contribution and its receiver/sender
orientation is inverted for ~93% of dev DT pairs (consistently in dev and eval, so the deployed r13 patch is still self-consistent). Here: nets from s_net in big blinds,
orientation taken from the role parquet (rs/ss, candidate-flow rule, 99.3% correct on dev DT), pot in big blinds. Surprisal logic (de-memorised v2/v1) is unchanged.
Usage: python x9_newfeats_fixed.py <role parquet with slot,h,rs,ss> <out parquet>"""
import numpy as np, pandas as pd, sys
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
d = pd.read_parquet(sys.argv[1])[["slot", "h", "rs", "ss"]].reset_index(drop=True)
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy", mmap_mode="r"); a_act = np.load(f"{D}/a_act.npy", mmap_mode="r"); a_amt = np.load(f"{D}/a_amount.npy", mmap_mode="r")
a_tc = np.load(f"{D}/a_to_call.npy", mmap_mode="r"); a_st = np.load(f"{D}/a_st.npy", mmap_mode="r"); a_pot = np.load(f"{D}/a_pot_before.npy", mmap_mode="r")
won = np.load(f"{D}/s_won.npy", mmap_mode="r"); net = np.load(f"{D}/s_net.npy", mmap_mode="r"); con = np.load(f"{D}/s_contrib.npy", mmap_mode="r"); bb = np.load(f"{D}/h_bb.npy").astype(float)
HS1 = np.load(f"{OUT}/HS1.npy", mmap_mode="r"); Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r")
Y = np.load(f"{OUT}/dec_Y.npy", mmap_mode="r"); P1 = np.load(f"{OUT}/dec_probs_v1.npy", mmap_mode="r"); P2 = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r")
N = int(off[-1]); in1 = np.zeros(N, bool); in1[np.random.RandomState(0).choice(N, 4_000_000, replace=False)] = True
in2 = np.zeros(N, bool); in2[np.random.RandomState(1).choice(N, 6_000_000, replace=False)] = True
FEAT = ["net_r", "net_s", "transfer", "pair_net_h", "both_vpip", "sd_pair", "r_won", "s_won_f", "sur_pass_s_max", "sur_pass_r_max", "sur_aggr_s_max", "sur_aggr_r_max", "sur_pass_max", "sur_all_sum",
        "str_fold_s", "str_fold_r", "str_pass_s_max", "str_pass_r_max", "mismatch_s_max", "mismatch_r_max", "s_fold_to_bet", "r_fold_to_bet", "s_call_pre", "r_aggr_any",
        "n_dec_s", "n_dec_r", "n_pass", "n_aggr", "pot_end_bb", "st_max", "strength_gap", "outs_folded"]
F = np.zeros((len(d), len(FEAT)))
for i, (h, rec, snd) in enumerate(zip(d.h.values, d.rs.values.astype(int), d.ss.values.astype(int))):
    ks = np.arange(off[h], off[h + 1]); kseat = np.asarray(a_seat[ks]); amt = np.asarray(a_amt[ks]).astype(float); tc = np.asarray(a_tc[ks]).astype(float)
    act = np.asarray(a_act[ks]); st = np.asarray(a_st[ks]); pot = np.asarray(a_pot[ks]).astype(float); B = bb[h]; wv = np.asarray(won[h])
    net_r = net[h, rec] / B; net_s = net[h, snd] / B; mem = (kseat == rec) | (kseat == snd); km = ks[mem]
    if not len(km): continue
    ksm = kseat[mem]; act_m = act[mem]; st_m = st[mem]; tc_m = tc[mem]; amt_m = amt[mem]
    y = np.asarray(Y[km]); p1 = np.asarray(P1[km])[np.arange(len(km)), y]; p2 = np.asarray(P2[km])[np.arange(len(km)), y]
    qc = np.where(~in2[km], p2, np.where(~in1[km], p1, np.nan)); sur = -np.log(np.clip(qc, 1e-6, 1)); ok = np.isfinite(sur)
    aggr = (act_m == 3) | (act_m == 4) | ((act_m == 5) & (amt_m > tc_m)); passive = ~aggr; is_fold = y == 0; S_ = ksm == snd; R_ = ksm == rec
    strength = np.asarray(HS1[h, st_m, ksm]).astype(float)
    mx = lambda v, m: float(np.nanmax(v[m & ok])) if (m & ok).any() else 0.0; mxs = lambda m: float(strength[m].max()) if m.any() else 0.0
    F[i] = [net_r, net_s, min(max(net_r, 0.0), max(-net_s, 0.0)), net_r + net_s, float(con[h, rec] > B and con[h, snd] > B), float(wv[rec] > 0 and wv[snd] > 0), float(wv[rec] > 0), float(wv[snd] > 0),
            mx(sur, passive & S_), mx(sur, passive & R_), mx(sur, aggr & S_), mx(sur, aggr & R_), mx(sur, passive), float(np.nansum(sur[ok])) if ok.any() else 0.0,
            mxs(is_fold & S_), mxs(is_fold & R_), mxs(passive & S_), mxs(passive & R_), mx(strength * sur, passive & S_), mx(strength * sur, passive & R_),
            float((S_ & is_fold & (tc_m > 0)).any()), float((R_ & is_fold & (tc_m > 0)).any()), float((S_ & (y == 2) & (st_m == 0)).any()), float((aggr & R_).any()),
            float(S_.sum()), float(R_.sum()), float(passive.sum()), float(aggr.sum()), float(pot[-1] / B if len(pot) else 0.0), float(st.max() if len(st) else 0),
            float(np.asarray(Pt[h, rec, 12]) - np.asarray(Pt[h, snd, 12])), float((np.asarray(Y[ks]) == 0).sum() - is_fold.sum())]
    if i % 20000 == 0: print(f"  {i}/{len(d)}", flush=True)
for j, c in enumerate(FEAT): d["f_" + c] = F[:, j]
d.drop(columns=["rs", "ss"]).to_parquet(sys.argv[2]); print("saved", sys.argv[2], d.shape)
