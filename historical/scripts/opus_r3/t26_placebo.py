"""Placebo control for the substitution statistic: card-independent scripted actions (CI raises, DT dumps) make ANY other hand
look better than the actor's own cards.  For each member decision (partner active) compare the partner's card block (q_P) with
the outsiders' card blocks (q_O, averaged over dealt outsiders).  F4 => BF_P >> BF_O; card-independent family => BF_P ~ BF_O.
Pair sets: eval members, B6, eval top-600 non-members with BF_hier > 5 (t17a); devsub11 labelled positives (calibration)."""
import numpy as np, pandas as pd, lightgbm as lgb, time
import m4b_policy_v2 as PV, aggmod as AG
exec(open("s83_hand_tilt.py").read().split("c38 = pd.read_parquet")[0].replace("@njit(cache=True)", "@njit"))
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
C_ = PV.ctx_counts(PV.nh, PV.off, PV.a_st, PV.a_seat, PV.a_tc, PV.s_player, PV.h_phase, PV.Y)
EX = np.zeros((PV.N, len(PV.FN2)), np.float32); PV.extra(PV.nh, PV.off, PV.a_st, PV.a_seat, PV.a_act, PV.a_amt, PV.a_amt_to, PV.a_tc, PV.a_pot, PV.s_player, PV.h_phase, C_, EX)
bst = lgb.Booster(model_file=f"{OUT}/policy_v2.txt")
HS1 = np.load(f"{OUT}/HS1.npy", mmap_mode="r"); HS2 = np.load(f"{OUT}/HS2.npy", mmap_mode="r"); CAT = np.load(f"{OUT}/CAT.npy", mmap_mode="r"); PF = np.asarray(Pt[:, :, 12]).astype(np.float32)
sp = np.load(f"{D}/s_player.npy", mmap_mode="r")
R3 = f"{OUT}/r3"; B = pd.read_parquet(f"{R3}/t17a_bf_eval.parquet")
b6 = open(f"{R3}/b6_promoted.txt").read().split(",")
sets = {"eval_member": B[B.member == True].slot.values, "eval_B6": B[B.pair_id.isin(b6)].slot.values,
        "eval_nonmem_top600_bf5": B[(B.member != True) & (~B.pair_id.isin(b6)) & (B.rk_r2j2m <= 600) & (B.bf_hier > 5)].slot.values}
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
o = pd.read_parquet(f"{OUT}/m15_v6ens_base_train_oof.parquet"); o = o[(o.src == "devsub11") & (o.y == 1)].copy()
lo = o.key // 12000; hi = o.key % 12000; o["slot"] = loc.pool.loc[lo].values * 900 + loc.local.loc[lo].values * 30 + loc.local.loc[hi].values
for f, g in o.groupby("fam"): sets[f"devsub_{f[:2]}"] = g.slot.values
def run(nm, slots, phase, mask):
    H, S, T, SL = PI.all_pair_hands(phase); m = np.isin(SL, slots) & (mask[H] == 1); H, S, T, SL = H[m], S[m], T[m], SL[m]
    rows = np.zeros((len(H) * 16, 5), np.int64); n = collect(H, S, T, off, a_seat, a_st, Y, rows); rows = rows[:n]
    r_, k, s, o_, st = rows.T; h = H[r_]; sl = SL[r_]
    order = np.argsort(k); Xo = np.hstack([np.asarray(PV.X1[k[order]]), EX[k[order]]]); inv = np.empty_like(order); inv[order] = np.arange(len(k)); Xo = Xo[inv]
    q0 = np.clip(np.asarray(P2[np.sort(k)])[inv], 1e-6, 1); q0 /= q0.sum(1, keepdims=True); y = Y[k]; ix = np.arange(len(k))
    def qswap(seat):
        X = Xo.copy(); X[:, 14] = HS1[h, st, seat]; X[:, 15] = HS2[h, st, seat]; X[:, 16] = CAT[h, st, seat]; X[:, 17] = PF[h, seat]
        q = np.clip(bst.predict(X, num_threads=2), 1e-6, 1); q /= q.sum(1, keepdims=True); return q[ix, y]
    rP = qswap(o_) / q0[ix, y]
    spH = np.asarray(sp[h]); rO_sum = np.zeros(len(k)); nO = np.zeros(len(k))
    for j in range(6):
        valid = (spH[:, j] >= 0) & (j != s) & (j != o_)
        if valid.sum() == 0: continue
        seat = np.where(valid, j, o_); rj = qswap(seat) / q0[ix, y]
        rO_sum += np.where(valid, rj, 0); nO += valid
    rO = rO_sum / np.maximum(nO, 1)
    a = np.where(st == 0, 0.362, 0.323)
    df = pd.DataFrame({"slot": sl, "lP": np.log((1 - a) + a * rP), "lO": np.log((1 - a) + a * rO)})
    g = df.groupby("slot").agg(bfP=("lP", "sum"), bfO=("lO", "sum"), n=("lP", "size")); g["D"] = g.bfP - g.bfO; g["set"] = nm
    log(nm, "pairs", len(g), "decisions", len(k)); return g.reset_index()
out = []
mk_eval = AG.hand_mask("eval"); mk_dev = AG.hand_mask("devsub", 11)
for nm, sl in sets.items():
    out.append(run(nm, sl, 1 if nm.startswith("eval") else 0, mk_eval if nm.startswith("eval") else mk_dev))
R = pd.concat(out, ignore_index=True).merge(B[["slot", "pair_id", "rk_r2j2m", "predicted_behavior", "bf_hier"]], on="slot", how="left")
R.to_parquet(f"{R3}/t26_placebo.parquet")
pd.set_option("display.width", 220)
print(R.groupby("set")[["bfP", "bfO", "D"]].describe(percentiles=[.1, .25, .5, .75, .9]).round(1).T.to_string())
print(R[R.set == "eval_nonmem_top600_bf5"].sort_values("D", ascending=False)[["pair_id", "rk_r2j2m", "predicted_behavior", "bf_hier", "bfP", "bfO", "D", "n"]].round(1).to_string(index=False))
print(R[R.set == "eval_member"].sort_values("D")[["pair_id", "rk_r2j2m", "bf_hier", "bfP", "bfO", "D", "n"]].head(12).round(1).to_string(index=False))
print(R[R.set == "eval_B6"][["pair_id", "rk_r2j2m", "bf_hier", "bfP", "bfO", "D", "n"]].round(1).to_string(index=False))
