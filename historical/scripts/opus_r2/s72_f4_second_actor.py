"""Fourth family: how does the SECOND member (B, acting after A preflop) respond, by A's first action and B's own strength?
Members vs controls.  Also B's response as a function of A's hand strength (does B also use A's cards?)."""
import numpy as np, pandas as pd
import pairindex as PI
from numba import njit
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"
exec(open("s71_f4_completion.py").read().split("base = pd.read_csv")[0].split("hidx = ")[0])
@njit(cache=True)
def seq(H, S, T, off, a_seat, a_st, Y, out):
    for r in range(len(H)):
        h = H[r]; ka = -1; kb = -1
        for k in range(off[h], off[h + 1]):
            if a_st[k] != 0: break
            if a_seat[k] == S[r] and ka < 0: ka = k
            if a_seat[k] == T[r] and kb < 0: kb = k
        if ka < 0 and kb < 0: out[r, 0] = -1; continue
        if kb < 0 or (ka >= 0 and ka < kb): f = ka; o = kb; out[r, 0] = 0
        else: f = kb; o = ka; out[r, 0] = 1
        out[r, 1] = Y[f]; out[r, 2] = Y[o] if o >= 0 else -1
        # number of raises between A's first action and B's first action (by anyone)
        nr = 0
        if o >= 0:
            for k in range(f + 1, o):
                if Y[k] == 3: nr += 1
        out[r, 3] = nr
def table(slots):
    H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, slots); H, S, T, SL = H[m], S[m], T[m], SL[m]
    O = np.zeros((len(H), 4), np.int64); seq(H, S, T, off, a_seat, a_st, Y, O); w = O[:, 0]
    X = pd.DataFrame({"who": w, "y1": O[:, 1], "y2": O[:, 2], "nr_between": O[:, 3]})
    X["eA"] = np.where(w == 0, pfeq[H, S], pfeq[H, T]); X["eB"] = np.where(w == 0, pfeq[H, T], pfeq[H, S])
    X = X[(X.who >= 0) & (X.y2 >= 0)].copy()
    X["a1"] = np.select([X.y1 == 0, X.y1 == 3], ["A_fold", "A_raise"], "A_call")
    X["bB"] = pd.cut(X.eB, [0, .45, .55, .65, 1.01], labels=["B<.45", "B.45-.55", "B.55-.65", "B>.65"])
    X["bA"] = pd.cut(X.eA, [0, .45, .55, .65, 1.01], labels=["A<.45", "A.45-.55", "A.55-.65", "A>.65"])
    X["b_fold"] = X.y2 == 0; X["b_raise"] = X.y2 == 3
    return X
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet")
base = pd.read_csv(f"{C}/r2j2_lgbcat2_p2comb_other_ev_on_r15.csv", dtype=str, usecols=["pair_id", "predicted_behavior"])
mids = set(base[base.predicted_behavior == "other_coordination"].pair_id)
Mt = table(c38[c38.pair_id.isin(mids)].slot.values); Ct = table(c38[c38.rk > 5000].sample(4000, random_state=1).slot.values)
pd.set_option("display.width", 220)
for nm, X in [("MEMBERS", Mt), ("CONTROL", Ct)]:
    print(f"==== {nm}: P(B folds | A's first action, B's own strength)   [n in brackets]")
    t = X.pivot_table(index="a1", columns="bB", values="b_fold", aggfunc="mean").round(2); n = X.pivot_table(index="a1", columns="bB", values="b_fold", aggfunc="size")
    print(t.astype(str) + " [" + n.astype(str) + "]")
    print(f"  {nm}: P(B folds | A raised, B strong>.65) by A's own strength:")
    G = X[(X.a1 == "A_raise") & (X.eB > .65)]
    print("   ", G.groupby("bA", observed=True).b_fold.agg(["mean", "size"]).round(2).T.to_dict())
    G = X[(X.a1 == "A_fold")]
    print(f"  {nm}: P(B raises | A folded) by A's strength (does B play A's cards?):", G.groupby("bA", observed=True).b_raise.agg(["mean", "size"]).round(3).T.to_dict())
