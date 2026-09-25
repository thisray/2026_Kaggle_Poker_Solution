"""Bayesian update of the fourth-family evidence-rule hypotheses from LB comparisons.
obs: 'A-B=delta' items (public S differences between two submissions that differ ONLY in the fourth-family evidence), e.g.
  OBS='c-first-CIr2c=0.00049,ND-c-first=0.0123,NDw-c-first=-0.0010'
Predicted delta = K * (AP_A - AP_B) under each hypothesis (s94 table), K = 0.2 * (public 4th-family pairs / public positives).
Posterior weights, posterior expected AP for every candidate, and suggested MIX weights (c13 components)."""
import numpy as np, pandas as pd, os, json
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
T = pd.read_csv(f"{OUT}/" + os.environ.get("TIN", "s94_unified_table.csv"), index_col=0)
PRIOR = {"HD": .10, "H2": .05, "H3": .05, "HH": .03, "DEVF": .07, "DEVA": .07, "HDxPW": .10, "HDxFLOP": .04, "DEVAxPW": .04,
         "HDx.5": .08, "DEVAx.5": .05, "DEVFx.5": .04, "M3C": .07, "M1C": .06, "UNIF": .15, "HBOTH": .05,
         "DEVFxPW": .05, "DEVFxPWx.5": .04, "DEVAxPWx.5": .04, "HDxPWx.5": .05, "HDxPWx.25": .03, "DEVFxPWx.25": .03,
         "HDxPAW": .05, "HDxPAWx.5": .04, "DEVxPAW": .04, "DEVxPAWx.5": .03, "HDxACW": .03,
         "HDxWSD": .05, "HDxWNOSD": .05, "AGGxPW": .05, "HDxWSDx.5": .03, "HDxWNOSDx.5": .03, "AGGxPWx.5": .03}
pr = pd.Series(PRIOR); pr = pr[pr.index.isin(T.index)]; pr = pr / pr.sum(); T = T.loc[pr.index]
OBS = [x for x in os.environ.get("OBS", "c-first-CIr2c=0.00049").split(",") if x]
SE = float(os.environ.get("SE", "0.0025"))
Ks = np.array([float(x) for x in os.environ.get("KS", "0.025,0.033,0.041").split(",")]); Kw = np.array([float(x) for x in os.environ.get("KW", "0.25,0.5,0.25").split(",")]); Kw = Kw / Kw.sum()
def parse(o):
    lhs, v = o.split("="); a, b = None, None
    for cand in sorted(T.columns, key=len, reverse=True):
        if lhs.startswith(cand + "-") and lhs[len(cand) + 1:] in T.columns: a, b = cand, lhs[len(cand) + 1:]; break
    assert a is not None, o
    return a, b, float(v)
obs = [parse(o) for o in OBS]
loglik = pd.Series(0.0, index=T.index)
for h in T.index:
    lk = 0.0
    for K, w in zip(Ks, Kw):
        l = 1.0
        for a, b, v in obs:
            pred = K * (T.loc[h, a] - T.loc[h, b]); l *= np.exp(-0.5 * ((v - pred) / SE) ** 2)
        lk += w * l
    loglik[h] = np.log(lk + 1e-300)
post = pr * np.exp(loglik - loglik.max()); post = post / post.sum()
# posterior over K marginalised over hypotheses
kpost = np.zeros(len(Ks))
for h in T.index:
    for j, (K, w) in enumerate(zip(Ks, Kw)):
        l = 1.0
        for a, b, v in obs: l *= np.exp(-0.5 * ((v - K * (T.loc[h, a] - T.loc[h, b])) / SE) ** 2)
        kpost[j] += pr[h] * w * l
kpost /= kpost.sum(); print("posterior over K:", dict(zip(Ks.round(4), kpost.round(3))))
print("observations:", obs)
print(pd.DataFrame({"prior": pr, "posterior": post}).round(3).sort_values("posterior", ascending=False).to_string())
EAP = (T.mul(post, axis=0)).sum(0).sort_values(ascending=False)
print("posterior expected AP@5 (fourth family) by candidate:"); print(EAP.round(3).to_string())
mix = {"HD": post[["HD", "HH"]].sum(), "H2": post[["H2", "H3"]].sum(), "DEVF": post[["DEVF", "DEVFx.5"]].sum(), "DEVA": post[["DEVA", "DEVAx.5"]].sum(),
       "HDPW": post[["HDxPW", "DEVAxPW"]].sum(), "HDFL": post["HDxFLOP"], "HDH": post["HDx.5"], "M3C": post["M3C"], "M1C": post["M1C"], "UNIF": post["UNIF"]}
if "HBOTH" in post.index: mix["BOTH"] = post["HBOTH"]
print("suggested WEIGHTS for c13:", ",".join(f"{k}={v:.3f}" for k, v in mix.items()))
json.dump({"obs": OBS, "posterior": post.to_dict(), "eap": EAP.to_dict(), "mix": {k: float(v) for k, v in mix.items()}}, open(f"{OUT}/s95_posterior.json", "w"), indent=1)
