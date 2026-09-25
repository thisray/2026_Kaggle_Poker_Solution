"""R3-F21: placebo D (partner cards vs outsider cards) on de-memorised policy ratios for the mid-rank candidates of t41.
Per decision: out of policy_v2's training rows -> v2 for q0 and every card swap; in v2's rows but not v1's -> v1 (26 base
features) for q0 and the swaps; in both -> skipped. D = BF_P - BF_O with a per-decision mixture a = 0.45.
Sets: eval non-member pairs ranked > 344 with clean M3 LLR > 5 (candidates); devsub11 pairs ranked > 344 with LLR > 5
(selected null); devsub11 labelled positives with LLR > 5 (card-independent known-family reference)."""
import numpy as np, pandas as pd, lightgbm as lgb, time
import m4b_policy_v2 as PV, aggmod as AG
exec(open("s83_hand_tilt.py").read().split("c38 = pd.read_parquet")[0].replace("@njit(cache=True)", "@njit"))
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
C_ = PV.ctx_counts(PV.nh, PV.off, PV.a_st, PV.a_seat, PV.a_tc, PV.s_player, PV.h_phase, PV.Y)
EX = np.zeros((PV.N, len(PV.FN2)), np.float32); PV.extra(PV.nh, PV.off, PV.a_st, PV.a_seat, PV.a_act, PV.a_amt, PV.a_amt_to, PV.a_tc, PV.a_pot, PV.s_player, PV.h_phase, C_, EX)
log("extra features")
b2 = lgb.Booster(model_file=f"{OUT}/policy_v2.txt"); b1 = lgb.Booster(model_file=f"{OUT}/policy_v1.txt")
P1 = np.load(f"{OUT}/dec_probs_v1.npy", mmap_mode="r")
N = PV.N; in2 = np.zeros(N, bool); in2[np.random.RandomState(1).choice(N, 6_000_000, replace=False)] = True
in1 = np.zeros(N, bool); in1[np.random.RandomState(0).choice(N, 4_000_000, replace=False)] = True
HS1 = np.load(f"{OUT}/HS1.npy", mmap_mode="r"); HS2 = np.load(f"{OUT}/HS2.npy", mmap_mode="r"); CAT = np.load(f"{OUT}/CAT.npy", mmap_mode="r"); PF = np.asarray(Pt[:, :, 12]).astype(np.float32)
sp = np.load(f"{D}/s_player.npy", mmap_mode="r"); R3 = f"{OUT}/r3"
M = pd.read_parquet(f"{R3}/t41_mid_clean.parquet")
E_ = M[(M.set == "eval") & (M.predicted_behavior != "other_coordination") & (M.rk > 344) & (M.llr > 5)]
V_ = M[(M.set == "devsub11")]
sets = {"eval_cand": E_.slot.values, "dev_null_sel": V_[(V_.rk > 344) & (V_.llr > 5) & (V_.y == 0)].slot.values,
        "dev_pos_sel": V_[(V_.y == 1) & (V_.llr > 5)].nlargest(60, "llr").slot.values}
log({k: len(v) for k, v in sets.items()})
A = 0.45
def run(nm, slots, phase, mask):
    H, S, T, SL = PI.all_pair_hands(phase); m = np.isin(SL, slots) & (mask[H] == 1); H, S, T, SL = H[m], S[m], T[m], SL[m]
    rows = np.zeros((len(H) * 16, 5), np.int64); n = collect(H, S, T, off, a_seat, a_st, Y, rows); rows = rows[:n]
    r_, k, s, o_, st = rows.T; h = H[r_]; sl = SL[r_]
    keep = ~(in2[k] & in1[k]); r_, k, s, o_, st, h, sl = r_[keep], k[keep], s[keep], o_[keep], st[keep], h[keep], sl[keep]
    use1 = in2[k]                                                     # v2 memorised this decision, v1 did not
    order = np.argsort(k); Xo = np.hstack([np.asarray(PV.X1[k[order]]), EX[k[order]]]); inv = np.empty_like(order); inv[order] = np.arange(len(k)); Xo = Xo[inv]
    q02 = np.clip(np.asarray(P2[np.sort(k)])[inv], 1e-6, 1); q02 /= q02.sum(1, keepdims=True)
    q01 = np.clip(np.asarray(P1[np.sort(k)])[inv], 1e-6, 1); q01 /= q01.sum(1, keepdims=True)
    y = Y[k]; ix = np.arange(len(k)); q0 = np.where(use1, q01[ix, y], q02[ix, y])
    def qswap(seat):
        X = Xo.copy(); X[:, 14] = HS1[h, st, seat]; X[:, 15] = HS2[h, st, seat]; X[:, 16] = CAT[h, st, seat]; X[:, 17] = PF[h, seat]
        q = np.empty(len(k))
        for model, msk, ncol in ((b2, ~use1, 38), (b1, use1, 26)):
            if msk.any():
                p = np.clip(model.predict(X[msk][:, :ncol], num_threads=2), 1e-6, 1); p /= p.sum(1, keepdims=True); q[msk] = p[np.arange(msk.sum()), y[msk]]
        return q
    rP = qswap(o_) / q0
    spH = np.asarray(sp[h]); rO_sum = np.zeros(len(k)); nO = np.zeros(len(k))
    for j in range(6):
        valid = (spH[:, j] >= 0) & (j != s) & (j != o_)
        if valid.sum() == 0: continue
        seat = np.where(valid, j, o_); rj = qswap(seat) / q0
        rO_sum += np.where(valid, rj, 0); nO += valid
    rO = rO_sum / np.maximum(nO, 1)
    df = pd.DataFrame({"slot": sl, "lP": np.log((1 - A) + A * rP), "lO": np.log((1 - A) + A * rO)})
    g = df.groupby("slot").agg(bfP=("lP", "sum"), bfO=("lO", "sum"), n=("lP", "size")); g["D"] = g.bfP - g.bfO; g["set"] = nm
    log(nm, "pairs", len(g), "decisions", len(k), "v1 share", round(float(use1.mean()), 3)); return g.reset_index()
out = []
mk_eval = AG.hand_mask("eval"); mk_dev = AG.hand_mask("devsub", 11)
for nm, sl in sets.items():
    out.append(run(nm, sl, 1 if nm.startswith("eval") else 0, mk_eval if nm.startswith("eval") else mk_dev))
R = pd.concat(out, ignore_index=True).merge(M[["slot", "set", "pair_id", "rk", "predicted_behavior", "llr", "fam"]].rename(columns={"set": "src"}), on="slot", how="left")
R["Dn"] = R.D / np.sqrt(R.n); R.to_parquet(f"{R3}/t42_placebo_clean.parquet")
pd.set_option("display.width", 230)
print(R.groupby("set")[["D", "Dn", "llr"]].describe(percentiles=[.1, .5, .9]).round(2).T.to_string())
print(R[R.set == "eval_cand"].sort_values("Dn", ascending=False)[["pair_id", "rk", "predicted_behavior", "llr", "bfP", "bfO", "D", "n", "Dn"]].round(2).to_string(index=False))
