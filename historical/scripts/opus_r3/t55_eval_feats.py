"""R3-E15b: the t52 evidence features for the EVAL candidate hands (R15 scored file: 4,000 pairs x 20 candidates).
Feature expressions are copied verbatim from t52_dtsp_features.py so dev-fitted weights transfer unchanged."""
import numpy as np, pandas as pd
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; RAW = f"{A_}/data/raw"; D = f"{OUT}/np"
import pairindex as PI
sc = pd.read_csv(f"{A_}/round15_campaign/scored_tabicl_rank_blend.csv")[["slot", "pair_id", "hand_id", "score"]]
ev = pd.read_csv(f"{RAW}/evaluation_pairs.csv"); pidx = pd.read_parquet(f"{D}/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
ev["pa"] = np.minimum(ev.player_1.map(pmap).values, ev.player_2.map(pmap).values)
ev["pb"] = np.maximum(ev.player_1.map(pmap).values, ev.player_2.map(pmap).values)
d = sc.merge(ev[["pair_id", "pa", "pb"]], on="pair_id")
hidx = pd.read_parquet(f"{D}/hand_index.parquet").set_index("hand_id").hi
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy", mmap_mode="r"); sp = np.load(f"{D}/s_player.npy", mmap_mode="r")
a_act = np.load(f"{D}/a_act.npy", mmap_mode="r"); a_amt = np.load(f"{D}/a_amount.npy", mmap_mode="r"); a_tc = np.load(f"{D}/a_to_call.npy", mmap_mode="r")
a_st = np.load(f"{D}/a_st.npy", mmap_mode="r"); a_pot = np.load(f"{D}/a_pot_before.npy", mmap_mode="r")
won = np.load(f"{D}/s_won.npy", mmap_mode="r"); HS1 = np.load(f"{OUT}/HS1.npy", mmap_mode="r"); Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r")
Y = np.load(f"{OUT}/dec_Y.npy", mmap_mode="r"); P1 = np.load(f"{OUT}/dec_probs_v1.npy", mmap_mode="r"); P2 = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r")
N = int(off[-1]); in1 = np.zeros(N, bool); in1[np.random.RandomState(0).choice(N, 4_000_000, replace=False)] = True
in2 = np.zeros(N, bool); in2[np.random.RandomState(1).choice(N, 6_000_000, replace=False)] = True
H, S_, T_, SL = PI.all_pair_hands(1)
need = np.isin(SL, d.slot.unique()); H, S_, T_, SL = H[need], S_[need], T_[need], SL[need]
W = np.asarray(won[H]); netS = np.zeros(len(H)); netT = np.zeros(len(H))
for i, h in enumerate(H):
    ks = np.arange(off[h], off[h + 1]); seats = np.asarray(a_seat[ks]); amt = np.asarray(a_amt[ks]).astype(float)
    cs = np.zeros(6); np.add.at(cs, seats, amt)
    netS[i] = W[i, S_[i]] - cs[S_[i]]; netT[i] = W[i, T_[i]] - cs[T_[i]]
pair_net = pd.DataFrame({"slot": SL, "nS": netS, "nT": netT}).groupby("slot").sum()
recv_is_T = (pair_net.nT > pair_net.nS).to_dict()
print(f"eval candidates {len(d)}, pairs {d.pair_id.nunique()}", flush=True)
FEAT = ["net_r", "net_s", "transfer", "pair_net_h", "both_vpip", "sd_pair", "r_won", "s_won_f",
        "sur_pass_s_max", "sur_pass_r_max", "sur_aggr_max", "sur_pass_max", "sur_all_sum",
        "str_fold_max", "str_pass_max", "mismatch_max", "s_fold_to_r", "s_call_r_raise", "r_aggr_after_s",
        "n_dec_s", "n_dec_r", "n_pass", "n_aggr", "pot_end", "st_max", "strength_gap", "outs_folded_pre"]
F = np.zeros((len(d), len(FEAT))); hh = hidx.loc[d.hand_id].values
for i, (h, pa_, pb_, sl) in enumerate(zip(hh, d.pa.values, d.pb.values, d.slot.values)):
    seats = np.asarray(sp[h]); sa = int(np.flatnonzero(seats == pa_)[0]); sb = int(np.flatnonzero(seats == pb_)[0])
    rec, snd = (sb, sa) if recv_is_T.get(sl, False) else (sa, sb)
    ks = np.arange(off[h], off[h + 1]); kseat = np.asarray(a_seat[ks]); amt = np.asarray(a_amt[ks]).astype(float)
    tc = np.asarray(a_tc[ks]).astype(float); act = np.asarray(a_act[ks]); st = np.asarray(a_st[ks]); pot = np.asarray(a_pot[ks]).astype(float)
    cs = np.zeros(6); np.add.at(cs, kseat, amt); wv = np.asarray(won[h])
    net_r = wv[rec] - cs[rec]; net_s = wv[snd] - cs[snd]
    mem = (kseat == rec) | (kseat == snd)
    km = ks[mem]; kseat_m = kseat[mem]; act_m = act[mem]; st_m = st[mem]; tc_m = tc[mem]; amt_m = amt[mem]
    y = np.asarray(Y[km]); p1 = np.asarray(P1[km])[np.arange(len(km)), y]; p2 = np.asarray(P2[km])[np.arange(len(km)), y]
    qc = np.where(~in2[km], p2, np.where(~in1[km], p1, np.nan)); sur = -np.log(np.clip(qc, 1e-6, 1))
    aggr = (act_m == 3) | (act_m == 4) | ((act_m == 5) & (amt_m > tc_m)); passive = ~aggr
    strength = np.asarray(HS1[h, st_m, kseat_m]).astype(float)
    is_fold = y == 0; ok = np.isfinite(sur)
    f = lambda m: float(np.nanmax(sur[m])) if (m & ok).any() else 0.0
    F[i] = [net_r, net_s, min(max(net_r, 0.0), max(-net_s, 0.0)), net_r + net_s,
            float((cs[rec] > 0) and (cs[snd] > 0)), float((wv[rec] > 0) and (wv[snd] > 0)), float(wv[rec] > 0), float(wv[snd] > 0),
            f(passive & (kseat_m == snd)), f(passive & (kseat_m == rec)), f(aggr), f(passive), float(np.nansum(sur[ok])) if ok.any() else 0.0,
            float(strength[is_fold].max()) if is_fold.any() else 0.0, float(strength[passive].max()) if passive.any() else 0.0,
            float(np.nanmax((strength * sur)[passive & ok])) if (passive & ok).any() else 0.0,
            float(((kseat_m == snd) & is_fold & (tc_m > 0)).any()), float(((kseat_m == snd) & (y == 2) & (st_m == 0)).any()),
            float((aggr & (kseat_m == rec)).any()), float((kseat_m == snd).sum()), float((kseat_m == rec).sum()),
            float(passive.sum()), float(aggr.sum()), float(pot[-1] if len(pot) else 0.0), float(st.max() if len(st) else 0),
            float(np.asarray(Pt[h, rec, 12]) - np.asarray(Pt[h, snd, 12])), float((np.asarray(Y[ks]) == 0).sum() - is_fold.sum())]
    if i % 20000 == 0: print(f"  {i}/{len(d)}", flush=True)
for j, c in enumerate(FEAT): d[c] = F[:, j]
d.to_parquet(f"{OUT}/r3/t55_eval_feats.parquet"); print("saved", d.shape)
