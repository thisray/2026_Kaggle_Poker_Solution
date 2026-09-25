"""Do KNOWN-family planted hands also use the partner's cards?  Dev candidate pools (r15, 20 per labelled positive pair):
per member decision (partner active) the substitution LR r = q_sub(y)/q0(y); per hand sum/max log r.  AUC evidence vs
non-evidence within pools, per family; re-rank test b + w * within-pair rank(sum log r), per fold."""
import numpy as np, pandas as pd, lightgbm as lgb, time
from sklearn.metrics import roc_auc_score
import m4b_policy_v2 as PV
exec(open("s83_hand_tilt.py").read().split("c38 = pd.read_parquet")[0].replace("@njit(cache=True)", "@njit"))
t0 = time.time()
C_ = PV.ctx_counts(PV.nh, PV.off, PV.a_st, PV.a_seat, PV.a_tc, PV.s_player, PV.h_phase, PV.Y)
EX = np.zeros((PV.N, len(PV.FN2)), np.float32); PV.extra(PV.nh, PV.off, PV.a_st, PV.a_seat, PV.a_act, PV.a_amt, PV.a_amt_to, PV.a_tc, PV.a_pot, PV.s_player, PV.h_phase, C_, EX)
bst = lgb.Booster(model_file=f"{OUT}/policy_v2.txt")
HS1 = np.load(f"{OUT}/HS1.npy", mmap_mode="r"); HS2 = np.load(f"{OUT}/HS2.npy", mmap_mode="r"); CAT = np.load(f"{OUT}/CAT.npy", mmap_mode="r"); PF = np.asarray(Pt[:, :, 12]).astype(np.float32)
A2 = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"
d = pd.read_parquet(f"{A2}/round11_scoped/dev_oof_aligned.parquet")[["slot", "hand_id", "fold", "ev", "m_p", "rs_blend"]]
tab = pd.read_csv(f"{A2}/round15_campaign/tabicl_cv/predictions.csv.gz")[["slot", "hand_id", "score"]].rename(columns={"score": "tab"}); d = d.merge(tab, on=["slot", "hand_id"])
hidx = pd.read_parquet(f"{D}/hand_index.parquet").set_index("hand_id").hi; d["h"] = hidx.loc[d.hand_id].values
d["b"] = 0.6 * d.groupby("slot").tab.rank(pct=True) + 0.4 * d.groupby("slot").rs_blend.rank(pct=True)
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
sp = np.load(f"{D}/s_player.npy", mmap_mode="r"); H = d.h.values; sl = d.slot.values
plo = mem[sl // 900, (sl % 900) // 30]; phi = mem[sl // 900, sl % 30]; spH = np.asarray(sp[H])
S = np.argmax(spH == plo[:, None], axis=1).astype(np.int64); T = np.argmax(spH == phi[:, None], axis=1).astype(np.int64)
rows = np.zeros((len(H) * 16, 5), np.int64); n = collect(H.astype(np.int64), S, T, off, a_seat, a_st, Y, rows); rows = rows[:n]
r_, k, s, o, st = rows.T; h = H[r_]
Xo = np.hstack([np.asarray(PV.X1[np.sort(k)]), EX[np.sort(k)]]); order = np.argsort(k); inv = np.empty_like(order); inv[order] = np.arange(len(k)); Xo = Xo[inv]
Xs = Xo.copy(); Xs[:, 14] = HS1[h, st, o]; Xs[:, 15] = HS2[h, st, o]; Xs[:, 16] = CAT[h, st, o]; Xs[:, 17] = PF[h, o]
q0 = np.clip(bst.predict(Xo, num_threads=2), 1e-6, 1); qs = np.clip(bst.predict(Xs, num_threads=2), 1e-6, 1); y = Y[k]
lr = np.log(qs[np.arange(len(y)), y] / q0[np.arange(len(y)), y]); print(f"[{time.time()-t0:.0f}s] decisions {len(lr)}", flush=True)
R = pd.DataFrame({"row": r_, "lr": lr}).groupby("row").lr.agg(["sum", "max", "size"])
d["slr"] = R["sum"].reindex(np.arange(len(d))).fillna(0).values; d["mlr"] = R["max"].reindex(np.arange(len(d))).fillna(0).values
lab = pd.read_csv("/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw/development_labels.csv")
fam = d.groupby("slot").size()  # placeholder
X4 = pd.read_parquet(f"{OUT}/t4_wrong_vs_hit.parquet")[["slot", "h", "fam"]]; d = d.merge(X4, on=["slot", "h"], how="left")
for f, g in d.groupby("fam"):
    print(f"{f:22s} AUC ev vs non within pools: sum logr {roc_auc_score(g.ev, g.slr):.3f}  max logr {roc_auc_score(g.ev, g.mlr):.3f}  | mean sum logr ev {g[g.ev == 1].slr.mean():.3f} non {g[g.ev == 0].slr.mean():.3f}")
def ap5(g, col):
    top = g.sort_values(col, ascending=False, kind="mergesort").ev.values[:5]; hits = 0; s_ = 0.0
    for i, e in enumerate(top):
        if e: hits += 1; s_ += hits / (i + 1)
    return s_ / min(5, int(g.m_p.iloc[0]))
base = d.groupby("slot").apply(lambda g: ap5(g, "b")); fam_s = d.groupby("slot").fam.first(); fold = d.groupby("slot").fold.first()
d["rs"] = d.groupby("slot").slr.rank(pct=True)
for w in (0.05, 0.1, 0.2, 0.3):
    d["b2"] = d.b + w * d.rs; new = d.groupby("slot").apply(lambda g: ap5(g, "b2")); dl = new - base
    print(f"w {w}: E {base.mean():.4f} -> {new.mean():.4f} ({dl.mean():+.4f}); by family {dl.groupby(fam_s).mean().round(4).to_dict()}; per fold {dl.groupby(fold).mean().round(4).tolist()}")
