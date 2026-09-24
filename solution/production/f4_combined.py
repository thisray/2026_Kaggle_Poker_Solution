"""s38: combined p2 + later-decision statistic. Independent confirmation for deep fourth-family candidates: the same information statistic computed ONLY on A's decisions
AFTER the first preflop decision (while B is still in the hand) -- disjoint from the p2 evidence.  Null-calibrated on dev U."""
import os
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
import pairindex as PI
OUT = os.environ["POKER_WORK_DIR"]; D = f"{OUT}/np"
DATA = Path(os.environ["POKER_DATA_DIR"])
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_act = np.load(f"{D}/a_act.npy")
Y = np.load(f"{OUT}/dec_Y.npy"); P2 = np.load(f"{OUT}/dec_probs_v2.npy")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
pfeq = np.asarray(Pt[:, :, PN.index("pf_eq_rand")]).astype(np.float32); mu = float(pfeq.mean())
@njit(cache=True)
def accum(H, S, T, SL, off, a_seat, a_act, Y, P2, pfeq, mu, ACC):
    for r in range(len(H)):
        h = H[r]
        for d in range(2):
            a = S[r] if d == 0 else T[r]; b = T[r] if d == 0 else S[r]
            e = pfeq[h, b] - mu; b_active = True; Rh = 0.0; Vh = 0.0; na = 0
            for k in range(off[h], off[h + 1]):
                s = a_seat[k]
                if s == b:
                    if a_act[k] == 0: b_active = False
                    continue
                if s != a: continue
                if not b_active: break
                na += 1
                if na == 1: continue                      # skip A's first decision (that is the p2 evidence)
                p0 = P2[k, 0]; p3 = P2[k, 3]; y = Y[k]
                Rh += ((1.0 if y == 3 else 0.0) - p3) - ((1.0 if y == 0 else 0.0) - p0); Vh += p3 * (1 - p3) + p0 * (1 - p0) + 2 * p0 * p3
            if Vh > 0:
                sl = SL[r]; ACC[sl, d, 0] += Rh * e; ACC[sl, d, 1] += Vh * e * e
out = {}
for ph in [0, 1]:
    H, S, T, SL = PI.all_pair_hands(ph); ACC = np.zeros((360000, 2, 2)); accum(H, S, T, SL, off, a_seat, a_act, Y, P2, pfeq, mu, ACC)
    z = ACC[:, :, 0] / np.sqrt(ACC[:, :, 1] + 1e-9); out[ph] = np.minimum(z[:, 0], z[:, 1])
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); lp = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
dv["slot"] = PI.pair_slot(dv.pool.values, lp.local.loc[dv.p_lo].values, lp.local.loc[dv.p_hi].values)
np.save(f"{OUT}/s38_q_later_dev.npy", out[0]); np.save(f"{OUT}/s38_q_later_eval.npy", out[1])
d23 = pd.read_parquet(f"{OUT}/s23_infoshare_dev.parquet"); d23["p2"] = np.minimum(d23.za0 - d23.zf0, d23.za1 - d23.zf1)
dd = dv[["slot", "label", "fam", "oof", "n"]].merge(d23[["slot", "p2"]], on="slot", how="left"); dd["q"] = out[0][dd.slot.values]
nul = dd[(dd.label == -1) & (dd.oof < 0.02) & (dd.n >= 38)]
mp, sp_, mq, sq = nul.p2.mean(), nul.p2.std(), nul.q.mean(), nul.q.std()
print(f"null p2 mean {mp:.3f} sd {sp_:.3f}; q_later mean {mq:.3f} sd {sq:.3f}; corr(p2,q) null {np.corrcoef(nul.p2.fillna(0), nul.q)[0,1]:.3f}")
def comb(p2, q): return ((p2 - mp) / sp_ + (q - mq) / sq) / np.sqrt(2)
dd["zc"] = comb(dd.p2.fillna(mp), dd.q)
thr = {a: np.quantile(dd.loc[nul.index, "zc"], 1 - a) for a in [1e-2, 1e-3, 1e-4]}
print("combined null thresholds:", {k: round(v, 3) for k, v in thr.items()})
for f in ["directed_transfer", "soft_play", "coordinated_isolation"]:
    g = dd[(dd.label == 1) & (dd.fam == f)]
    print(f"dev {f[:2]} positives n {len(g)}: frac zc > thr(1e-3) {(g.zc > thr[1e-3]).mean():.3f}  > thr(1e-4) {(g.zc > thr[1e-4]).mean():.3f}")
from late_stage import _slot_table
evaluation = _slot_table(Path(OUT), DATA, "evaluation")[["slot", "pair_id"]]
local = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
risk = pd.read_parquet(f"{OUT}/m1_eval_scores.parquet")
risk["slot"] = PI.pair_slot(
    risk.pool.to_numpy(),
    local.local.loc[risk.p_lo].to_numpy(),
    local.local.loc[risk.p_hi].to_numpy(),
)
risk = evaluation.merge(risk[["slot", "score"]], on="slot", validate="one_to_one")
e = pd.read_parquet(f"{OUT}/s23_infoshare_eval.parquet").merge(risk, on="slot")
e = e.rename(columns={"score": "risk_score"})
e = e.sort_values(["risk_score", "pair_id"], ascending=[False, True]).reset_index(drop=True); e["rk"] = np.arange(1, len(e) + 1)
e["p2"] = np.minimum(e.za0 - e.zf0, e.za1 - e.zf1); e["q"] = out[1][e.slot.values]; e["zc"] = comb(e.p2, e.q)
done = (e.rk <= 600) & (e.p2 > 2.5)
for lo, hi in [(0, 450), (450, 600), (600, 1000), (1000, 2000), (2000, 5000), (5000, 120000)]:
    m = (e.rk > lo) & (e.rk <= hi)
    print(f"({lo},{hi}] zc>thr(1e-3): {int((m & (e.zc > thr[1e-3])).sum())} (null exp {1e-3 * m.sum():.1f}); of which not in r2c set: {int((m & (e.zc > thr[1e-3]) & ~done).sum())};  zc>thr(1e-4): {int((m & (e.zc > thr[1e-4])).sum())} (exp {1e-4 * m.sum():.2f})")
cand = e[(e.rk <= 1000) & ~done & (e.zc > thr[1e-3])].sort_values("zc", ascending=False)
print(cand[["pair_id", "rk", "risk_score", "p2", "q", "zc", "n_is"]].round(3).to_string())
e[["slot", "pair_id", "rk", "p2", "q", "zc"]].to_parquet(f"{OUT}/s38_combined_eval.parquet")
