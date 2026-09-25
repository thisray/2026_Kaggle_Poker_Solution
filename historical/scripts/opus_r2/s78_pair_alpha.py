"""Pair-specific activity rate for the fourth family (latent-activity idea from Round16): per pair MAP-EM of alpha_p for the
first-actor decisions (global Q_A, pi0_A from s73) with a Beta prior centred on the global alpha; the responder's rate is
scaled by alpha_B/alpha_A.  Reports heterogeneity and how much the dense-hypothesis decoders change."""
import numpy as np, pandas as pd, pickle
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"
Mx = pd.read_parquet(f"{OUT}/s73_f4_mech2.parquet"); S = pickle.load(open(f"{OUT}/s73_f4_sim.pkl", "rb"))
aA, aB, QA, QB, pi0A, pi0B = S["aA"], S["aB"], S["QA"], S["QB"], S["pi0A"], S["pi0B"]
qa = QA[0, Mx.oB.values, Mx.a1.values]; pa = pi0A[Mx.oA.values, Mx.a1.values]
hb = (Mx.a2 >= 0).values
qb = np.where(hb, QB[Mx.rb.values, Mx.oA.values, np.maximum(Mx.a2.values, 0)], 1.0); pb = np.where(hb, pi0B[Mx.rb.values, Mx.oB.values, np.maximum(Mx.a2.values, 0)], 1.0)
Mx["qa"], Mx["pa"], Mx["qb"], Mx["pb"] = qa, pa, qb, pb
KAPPA = 20.0; a0 = aA * KAPPA; b0 = (1 - aA) * KAPPA
rows = []
for sl, G in Mx.groupby("slot"):
    al = aA
    for it in range(50):
        r = al * G.qa / (al * G.qa + (1 - al) * G.pa)
        al_new = (r.sum() + a0 - 1) / (len(G) + a0 + b0 - 2)
        if abs(al_new - al) < 1e-6: break
        al = al_new
    rows.append((sl, al, len(G), float(np.log(al * G.qa + (1 - al) * G.pa).sum() - np.log(G.pa).sum())))
PA = pd.DataFrame(rows, columns=["slot", "alpha_p", "n", "llr"]); print(PA.alpha_p.describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_dict())
# unpooled MLE for heterogeneity check (kappa -> 0)
rows = []
for sl, G in Mx.groupby("slot"):
    al = aA
    for it in range(200):
        r = al * G.qa / (al * G.qa + (1 - al) * G.pa); al_new = r.mean()
        if abs(al_new - al) < 1e-7: break
        al = al_new
    rows.append(al)
mle = np.array(rows); print("unpooled MLE alpha_p quantiles:", np.round(np.quantile(mle, [.1, .25, .5, .75, .9]), 3), " sd", round(mle.std(), 3))
# expected sd of MLE under homogeneity (parametric bootstrap from the global model)
rng = np.random.default_rng(0); sims = []
for sl, G in list(Mx.groupby("slot")):
    # simulate actions from the fitted mixture given own/partner bins, then re-estimate alpha
    p_mix = (1 - aA) * pi0A[G.oA.values] + aA * QA[0, G.oB.values]
    acts = np.array([rng.choice(3, p=p / p.sum()) for p in p_mix])
    q_ = QA[0, G.oB.values, acts]; p_ = pi0A[G.oA.values, acts]; al = aA
    for it in range(200):
        r = al * q_ / (al * q_ + (1 - al) * p_); al_new = r.mean()
        if abs(al_new - al) < 1e-7: break
        al = al_new
    sims.append(al)
print("MLE sd expected under a homogeneous alpha:", round(np.std(sims), 3))
Mx = Mx.merge(PA[["slot", "alpha_p"]], on="slot")
Mx["P1p"] = Mx.alpha_p * Mx.qa / (Mx.alpha_p * Mx.qa + (1 - Mx.alpha_p) * Mx.pa)
abp = Mx.alpha_p * aB / aA
Mx["P2p"] = np.where(hb, abp * Mx.qb / (abp * Mx.qb + (1 - abp) * Mx.pb), 0.0)
Mx.to_parquet(f"{OUT}/s78_f4_pair_alpha.parquet"); PA.to_parquet(f"{OUT}/s78_pair_alpha_table.parquet")
print("corr(P1, P1p):", round(np.corrcoef(Mx.P1, Mx.P1p)[0, 1], 3))
