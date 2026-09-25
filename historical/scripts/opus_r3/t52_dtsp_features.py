"""R3-E13: role- and strength-conditioned evidence features for DT / SP (dev, official-metric E@5).

Hypothesis: the R15 evidence features carry the surprisal of AGGRESSIVE actions (S_sur_aggr_act_mx) but not of PASSIVE
ones. A planted directed transfer is a card-independent call/fold by the sender, and soft play is a card-independent
check/call by the member holding a strong hand: both are passive actions that the normal policy would rarely take with
those cards. Policy surprisal is taken de-memorised (policy_v2 where it did not train on the decision, else policy_v1,
else missing).

Evaluation: dev candidates of labelled positive pairs (the same table and blend as t3/t45), pool folds already in the
table, LightGBM binary on `ev`, compared with the deployed blend by per-family E@5 (mean AP@5 over pairs, official
denominator min(5, #true)). Reported for the model alone and for a rank blend with the deployed score.
"""
import numpy as np, pandas as pd, lightgbm as lgb, json
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; RAW = f"{A_}/data/raw"; D = f"{OUT}/np"
d = pd.read_parquet(f"{A_}/round11_scoped/dev_oof_aligned.parquet")[["slot", "hand_id", "fold", "ev", "rs_blend"]]
tab = pd.read_csv(f"{A_}/round15_campaign/tabicl_cv/predictions.csv.gz")[["slot", "hand_id", "score"]].rename(columns={"score": "tab"})
d = d.merge(tab, on=["slot", "hand_id"]); g_ = d.groupby("slot")
d["r_tab"] = g_.tab.rank(pct=True); d["r_rs"] = g_.rs_blend.rank(pct=True); d["b"] = 0.6 * d.r_tab + 0.4 * d.r_rs
hidx = pd.read_parquet(f"{D}/hand_index.parquet").set_index("hand_id").hi
pidx = pd.read_parquet(f"{D}/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
lab = pd.read_csv(f"{RAW}/development_labels.csv"); lab = lab[lab.label == 1].copy()
a = lab.player_1.map(pmap).values; b_ = lab.player_2.map(pmap).values; lo = np.minimum(a, b_); hi = np.maximum(a, b_)
lab["slot"] = loc.pool.loc[lo].values * 900 + loc.local.loc[lo].values * 30 + loc.local.loc[hi].values; lab["pa"] = lo; lab["pb"] = hi
d = d.merge(lab[["slot", "pa", "pb", "behavior_family"]], on="slot")
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy", mmap_mode="r"); sp = np.load(f"{D}/s_player.npy", mmap_mode="r")
a_act = np.load(f"{D}/a_act.npy", mmap_mode="r"); a_amt = np.load(f"{D}/a_amount.npy", mmap_mode="r"); a_tc = np.load(f"{D}/a_to_call.npy", mmap_mode="r")
a_st = np.load(f"{D}/a_st.npy", mmap_mode="r"); a_pot = np.load(f"{D}/a_pot_before.npy", mmap_mode="r")
won = np.load(f"{D}/s_won.npy", mmap_mode="r"); HS1 = np.load(f"{OUT}/HS1.npy", mmap_mode="r"); Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r")
Y = np.load(f"{OUT}/dec_Y.npy", mmap_mode="r"); P1 = np.load(f"{OUT}/dec_probs_v1.npy", mmap_mode="r"); P2 = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r")
N = int(off[-1]); in1 = np.zeros(N, bool); in1[np.random.RandomState(0).choice(N, 4_000_000, replace=False)] = True
in2 = np.zeros(N, bool); in2[np.random.RandomState(1).choice(N, 6_000_000, replace=False)] = True
import pairindex as PI
# pair-level direction: net chips of each member over ALL shared dev-phase hands (unsupervised, no labels)
H, S_, T_, SL = PI.all_pair_hands(0)
need = np.isin(SL, d.slot.unique()); H, S_, T_, SL = H[need], S_[need], T_[need], SL[need]
netS = np.zeros(len(H)); netT = np.zeros(len(H))
W = np.asarray(won[H]); ix = np.arange(len(H))
contrib = {}
for i, h in enumerate(H):
    ks = np.arange(off[h], off[h + 1]); seats = np.asarray(a_seat[ks]); amt = np.asarray(a_amt[ks]).astype(float)
    cs = np.zeros(6)
    np.add.at(cs, seats, amt)
    contrib[i] = cs
    netS[i] = W[i, S_[i]] - cs[S_[i]]; netT[i] = W[i, T_[i]] - cs[T_[i]]
pair_net = pd.DataFrame({"slot": SL, "nS": netS, "nT": netT}).groupby("slot").sum()
recv_is_T = (pair_net.nT > pair_net.nS).to_dict()          # fixed receiver guess per pair (member with the higher total net)
print(f"dev candidates {len(d)}, positive pairs {d.slot.nunique()}, true evidence {int(d.ev.sum())}", flush=True)

FEAT = ["net_r", "net_s", "transfer", "pair_net_h", "both_vpip", "sd_pair", "r_won", "s_won_f",
        "sur_pass_s_max", "sur_pass_r_max", "sur_aggr_max", "sur_pass_max", "sur_all_sum",
        "str_fold_max", "str_pass_max", "mismatch_max", "s_fold_to_r", "s_call_r_raise", "r_aggr_after_s",
        "n_dec_s", "n_dec_r", "n_pass", "n_aggr", "pot_end", "st_max", "strength_gap", "outs_folded_pre"]
F = np.zeros((len(d), len(FEAT)))
hh = hidx.loc[d.hand_id].values
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
    is_fold = y == 0
    ok = np.isfinite(sur)
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
for j, c in enumerate(FEAT): d[c] = F[:, j]
d.to_parquet(f"{OUT}/r3/t52_dev_feats.parquet")

def ap5(g, col):
    top = g.sort_values(col, ascending=False, kind="mergesort").head(5).ev.values; hits = 0; s = 0.0
    for i, e in enumerate(top):
        if e: hits += 1; s += hits / (i + 1)
    return s / min(5, int(g.ev.sum()))
params = dict(objective="binary", learning_rate=0.04, num_leaves=7, min_data_in_leaf=60, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=30, verbose=-1, num_threads=2)
res = {}
for fam in ("directed_transfer", "soft_play", "coordinated_isolation"):
    sub = d[d.behavior_family == fam].copy()
    for name, X in (("new", FEAT), ("new+base", FEAT + ["b", "r_tab", "r_rs"])):
        p = np.zeros(len(sub))
        for seed in (3, 7, 11):
            params["seed"] = seed; pp = np.zeros(len(sub))
            for fo in sorted(sub.fold.unique()):
                tr, va = sub.fold != fo, sub.fold == fo
                m = lgb.train(params, lgb.Dataset(sub.loc[tr, X], sub.loc[tr, "ev"]), num_boost_round=300)
                pp[va.values] = m.predict(sub.loc[va, X])
            p += pp / 3
        sub[f"p_{name}"] = p; sub[f"rk_{name}"] = sub.groupby("slot")[f"p_{name}"].rank(pct=True)
        sub[f"mix_{name}"] = 0.5 * sub[f"rk_{name}"] + 0.5 * sub.groupby("slot").b.rank(pct=True)
    e = {k: float(sub.groupby("slot").apply(lambda g: ap5(g, k)).mean()) for k in ("b", "p_new", "mix_new", "p_new+base", "mix_new+base")}
    res[fam] = dict(pairs=int(sub.slot.nunique()), **{k: round(v, 4) for k, v in e.items()})
    print(fam, json.dumps(res[fam]), flush=True)
    m = lgb.train(params, lgb.Dataset(sub[FEAT], sub.ev), num_boost_round=300)
    print("   top gains:", pd.Series(m.feature_importance("gain"), index=FEAT).sort_values(ascending=False).head(8).round(0).to_dict(), flush=True)
json.dump(res, open(f"{OUT}/r3/t52_dtsp.json", "w"), indent=1)
