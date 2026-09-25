"""CI and SP: evidence vs high-score non-evidence hands before the last evidence — sequence-level descriptors."""
import numpy as np, pandas as pd, json
from numba import njit
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D_ = f"{OUT}/np"
M = pd.read_parquet(f"{OUT}/m16_handscores.parquet")
q999 = np.quantile(M.s[(M.phase == 0) & (~M.pos)], 0.999)
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); members = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): members[pool, g.local.values] = g.player_gi.values
L = lambda x: np.load(f"{D_}/{x}.npy")
sp = L("s_player"); off = L("a_off"); a_st = L("a_st"); a_seat = L("a_seat"); a_act = L("a_act"); a_amt = L("a_amount"); a_tc = L("a_to_call")
net = L("s_net"); bb = L("h_bb"); nsd = L("h_nsd"); board = L("h_board"); HS1 = np.load(f"{OUT}/HS1.npy", mmap_mode="r"); eqla = np.load(f"{OUT}/act_eqla_v1.npy", mmap_mode="r"); eqm = np.load(f"{OUT}/act_eqm_v1.npy", mmap_mode="r")
probs = np.load(f"{OUT}/dec_probs_v1.npy", mmap_mode="r")
@njit(cache=True)
def describe(h, A, B, off, a_st, a_seat, a_act, a_amt, a_tc, eqla, eqm, pa_arr, out):
    first_aggr = -1; n_aggr_A = 0; n_aggr_B = 0; n_aggr_O = 0; o_fold_after_pair = 0; A_fold_to_B = 0; B_fold_to_A = 0
    pair_reraise = 0; last_aggr = -1; cur = -1; o_entered = 0; max_st = 0; A_last_fold_eq = -1.0; B_last_fold_eq = -1.0
    checks_hu = 0; aggr_hu = 0; active = np.ones(6); outs_active_at_reraise = 0; passive_high_pa = 0.0; weak_aggr = 0.0
    for k in range(off[h], off[h + 1]):
        st = a_st[k]
        if st != cur:
            cur = st; last_aggr = -1
        if st > max_st: max_st = st
        i = a_seat[k]; act = a_act[k]; tc = a_tc[k]; amt = a_amt[k]
        aggr = act == 3 or act == 4 or (act == 5 and amt > tc)
        nact = 0
        for s in range(6): nact += active[s]
        if aggr:
            if first_aggr < 0: first_aggr = 0 if i == A else (1 if i == B else 2)
            if i == A: n_aggr_A += 1
            elif i == B: n_aggr_B += 1
            else: n_aggr_O += 1
            if (i == A and last_aggr == B) or (i == B and last_aggr == A):
                pair_reraise += 1
                c = 0
                for s in range(6):
                    if s != A and s != B and active[s] == 1: c += 1
                outs_active_at_reraise = max(outs_active_at_reraise, c)
            if (i == A or i == B) and pa_arr[k] < 0.2: weak_aggr += 1
        if act == 0:
            if i != A and i != B and (last_aggr == A or last_aggr == B): o_fold_after_pair += 1
            if i == A and last_aggr == B: A_fold_to_B += 1; A_last_fold_eq = eqla[k]
            if i == B and last_aggr == A: B_fold_to_A += 1; B_last_fold_eq = eqla[k]
            active[i] = 0
        if (act == 2 or (act == 5 and amt <= tc)) and i != A and i != B and st == 0: o_entered += 1
        if nact == 2 and active[A] == 1 and active[B] == 1 and (i == A or i == B):
            if act == 1: checks_hu += 1; passive_high_pa += pa_arr[k]
            if aggr: aggr_hu += 1
        if aggr: last_aggr = i
    out[0] = first_aggr; out[1] = n_aggr_A + n_aggr_B; out[2] = n_aggr_O; out[3] = o_fold_after_pair; out[4] = A_fold_to_B + B_fold_to_A
    out[5] = pair_reraise; out[6] = o_entered; out[7] = max_st; out[8] = max(A_last_fold_eq, B_last_fold_eq); out[9] = checks_hu; out[10] = aggr_hu
    out[11] = outs_active_at_reraise; out[12] = passive_high_pa; out[13] = weak_aggr
names = ["first_aggr_role", "pair_aggr", "o_aggr", "o_fold_after_pair", "pair_fold_to_partner", "pair_reraise", "o_entered_pre", "max_street", "fold_to_partner_eq", "checks_hu", "aggr_hu", "outs_at_reraise", "hu_check_pa_sum", "weak_aggr_pair"]
for fam in ["coordinated_isolation", "soft_play"]:
    F = M[(M.pos) & (M.fam == fam)].copy()
    last_ev = F[F.ev].groupby("sl").ts.max(); F["before_last"] = F.ts < F.sl.map(last_ev)
    E = F[F.ev]; X = F[(~F.ev) & F.before_last & (F.s > q999)]; N = F[(~F.ev) & (F.s < 0.05)].sample(3000, random_state=0)
    res = {}
    pa = np.asarray(probs[:, 3])
    for nm, G in [("EVIDENCE", E), ("X_nonev_hi", X), ("normal", N)]:
        sl = G.sl.values; hh = G.h.values
        plo = members[sl // 900, (sl % 900) // 30]; phi = members[sl // 900, sl % 30]
        sa = np.argmax(sp[hh] == plo[:, None], axis=1); sb = np.argmax(sp[hh] == phi[:, None], axis=1)
        arr = np.zeros((len(G), len(names)))
        for r in range(len(G)):
            describe(hh[r], sa[r], sb[r], off, a_st, a_seat, a_act, a_amt, a_tc, np.asarray(eqla[off[hh[r]]:off[hh[r] + 1]]) if False else eqla, eqm, pa, arr[r])
        df = pd.DataFrame(arr, columns=names)
        res[nm] = df.mean(); res[nm + "_first_aggr=pair"] = pd.Series({"frac": ((df.first_aggr_role == 0) | (df.first_aggr_role == 1)).mean()})
        if nm == "EVIDENCE": dfE = df
        if nm == "X_nonev_hi": dfX = df
    print(f"===== {fam}: evidence {len(E)}  X {len(X)}")
    print(pd.DataFrame({k: v for k, v in res.items() if not k.endswith("pair")}).round(3).to_string())
    print("first aggressor is pair member: evidence", round(((dfE.first_aggr_role == 0) | (dfE.first_aggr_role == 1)).mean(), 3), " X", round(((dfX.first_aggr_role == 0) | (dfX.first_aggr_role == 1)).mean(), 3))
    print("ended preflop (max_street=0): evidence", round((dfE.max_street == 0).mean(), 3), " X", round((dfX.max_street == 0).mean(), 3))
