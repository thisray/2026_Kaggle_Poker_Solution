"""R4-Z2: is the fourth family a SPARSE script of fully partner-card-driven hands on top of a dense background (as in the known families, whose background
information sharing is not evidence), or one dense hierarchical process (the R3 M0/M3 view)?
Pooled fixed-effect mixtures over the member hands (de-memorised ratios r as in t37 OOT=1 RV1=1):
  A: (1-rho) + rho * prod_d (1 - s + s r_d)                                   [R3 M0]
  B: (1-rb-rf) + rb * prod_d (1 - sb + sb r_d) + rf * prod_d (1 - sf + sf r_d)   [background + planted], sf free
Reports log-likelihoods, fitted parameters and the expected number of component-f hands per pair."""
import numpy as np, pandas as pd, sys, os
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917")
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; C = f"{OUT}/r2_candidates"; R3 = f"{OUT}/r3"
base = pd.read_csv(f"{C}/r2n_NDw_on_r2j2m.csv", dtype=str).set_index("pair_id"); oth = base.index[base.predicted_behavior == "other_coordination"].tolist()
e85 = pd.read_parquet(f"{OUT}/s85_eval_bf.parquet")[["slot", "pair_id"]].set_index("pair_id").slot; slots = e85.loc[oth].values
rows = pd.read_parquet(f"{R3}/t17_rows_eval.parquet", columns=["k", "h", "st", "slot", "r"]); rows = rows[rows.slot.isin(slots)]
NDEC = len(np.load(f"{OUT}/dec_Y.npy", mmap_mode="r")); intrain = np.zeros(NDEC, bool); intrain[np.random.RandomState(1).choice(NDEC, 6_000_000, replace=False)] = True
v1 = pd.read_parquet(f"{R3}/t40_v1_ratio.parquet")[["k", "slot", "r_v1", "in_v1"]]; rows = rows.merge(v1, on=["k", "slot"], how="left")
use1 = intrain[rows.k.values] & ~rows.in_v1.fillna(True).values.astype(bool); both = intrain[rows.k.values] & rows.in_v1.fillna(True).values.astype(bool)
rows.loc[use1, "r"] = rows.r_v1[use1]; rows.loc[both, "r"] = 1.0
H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, slots); G = pd.DataFrame({"slot": SL[m], "h": H[m]}).drop_duplicates().reset_index(drop=True); G["hid"] = np.arange(len(G))
rows = rows.merge(G, on=["slot", "h"]); hid = rows.hid.values; lr = np.log(rows.r.values.clip(1e-12, None)); pre = rows.st.values == 0; NH = len(G)
print(f"{len(oth)} member pairs, {NH} co-seated hands, {len(rows)} member decisions; hands per pair {NH / len(oth):.1f}", flush=True)
def hand_ll(s_pre, s_post):
    s = np.where(pre, s_pre, s_post); v = np.log1p(-s + s * np.exp(lr)); out = np.zeros(NH); np.add.at(out, hid, v); return out
def nllA(th):
    rho, sp, so = expit(th); l = hand_ll(sp, so); return -np.logaddexp(np.log1p(-rho), np.log(rho) + l).sum()
def nllB(th):
    w = np.exp(th[:3] - logsumexp(th[:3])); sb_p, sb_o, sf_p, sf_o = expit(th[3:7]); lb = hand_ll(sb_p, sb_o); lf = hand_ll(sf_p, sf_o)
    return -logsumexp(np.stack([np.log(w[0]) + np.zeros(NH), np.log(w[1]) + lb, np.log(w[2]) + lf]), axis=0).sum()
a = minimize(nllA, [0.9, 0.0, -0.3], method="L-BFGS-B"); rho, sp, so = expit(a.x); print(f"A: LL {-a.fun:.1f} rho {rho:.3f} s_pre {sp:.3f} s_post {so:.3f}")
best = None
for init in ([0.0, 1.0, -1.5, -1.0, -1.2, 3.0, 3.0], [0.5, 0.5, -2.5, -0.5, -0.8, 2.0, 2.0], [0.0, 0.0, 0.0, -2.0, -2.0, 1.0, 1.0]):
    b = minimize(nllB, init, method="L-BFGS-B");
    if best is None or b.fun < best.fun: best = b
w = np.exp(best.x[:3] - logsumexp(best.x[:3])); sb_p, sb_o, sf_p, sf_o = expit(best.x[3:7])
print(f"B: LL {-best.fun:.1f} (gain {a.fun - best.fun:+.1f} for 4 extra parameters) weights normal {w[0]:.3f} background {w[1]:.3f} planted {w[2]:.3f}; s_b pre/post {sb_p:.3f}/{sb_o:.3f}; s_f pre/post {sf_p:.3f}/{sf_o:.3f}")
print(f"   expected hands per pair: background {w[1] * NH / len(oth):.1f}, planted-component {w[2] * NH / len(oth):.1f}")
lb = hand_ll(sb_p, sb_o); lf = hand_ll(sf_p, sf_o); post = np.exp(np.log(w[2]) + lf - logsumexp(np.stack([np.log(w[0]) + np.zeros(NH), np.log(w[1]) + lb, np.log(w[2]) + lf]), axis=0))
G["post_f"] = post; G["nd"] = np.bincount(hid, minlength=NH); per = G.groupby("slot").post_f.sum()
print("   posterior mass of the planted component per pair: quantiles", per.quantile([.1, .25, .5, .75, .9]).round(2).tolist(), "| hands with post_f > 0.8:", int((post > 0.8).sum()), f"({(post > 0.8).sum() / len(oth):.2f} per pair)")
G.to_parquet(f"{OUT}/r4/z2_f4_hand_post.parquet")
