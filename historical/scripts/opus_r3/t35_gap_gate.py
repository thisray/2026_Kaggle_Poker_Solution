"""R3-F14: does the substitution switch on when the PARTNER's cards are stronger than the actor's own ("play the better hand")?

Per-decision mixture L_d = (1 - a_d) + a_d * r_d on the t17 rows (r_d = P(y | partner cards) / P(y | own cards)), with
a_d = sigmoid(c_street + b * gap_d) and gap_d = partner minus own card strength at the decision street.
Compared on the current other_coordination members and on control pairs (top-ranked known-family predictions, and
mid-rank unlabeled pairs). A 'better hand' mechanism predicts b >> 0 on members and nothing on controls; the binned
activity profile shows the shape directly."""
import os, sys, json
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; C = f"{OUT}/r2_candidates"; R3 = f"{OUT}/r3"
HS1 = np.load(f"{OUT}/HS1.npy", mmap_mode="r"); HS2 = np.load(f"{OUT}/HS2.npy", mmap_mode="r"); CAT = np.load(f"{OUT}/CAT.npy", mmap_mode="r")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r")
base = pd.read_csv(f"{C}/r5_subh.csv", dtype=str); base["rk"] = base.risk_score.astype(float).rank(ascending=False, method="first")
e85 = pd.read_parquet(f"{OUT}/s85_eval_bf.parquet")[["slot", "pair_id"]]; base = base.merge(e85, on="pair_id")
groups = {"members": base[base.predicted_behavior == "other_coordination"].slot.values,
          "known_top600": base[(base.predicted_behavior != "other_coordination") & (base.rk <= 600)].slot.values,
          "mid_1500_3000": base[(base.rk > 1500) & (base.rk <= 3000)].slot.values}
rows = pd.read_parquet(f"{R3}/t17_rows_eval.parquet", columns=["h", "st", "s", "o", "slot", "r"])
allsl = np.concatenate(list(groups.values())); rows = rows[rows.slot.isin(allsl)].reset_index(drop=True)
h, st, s, o = rows.h.values, rows.st.values, rows.s.values, rows.o.values
feat = {}
hs1 = np.asarray(HS1[h, st, o]) - np.asarray(HS1[h, st, s]); feat["hs1"] = hs1
feat["hs2"] = np.asarray(HS2[h, st, o]) - np.asarray(HS2[h, st, s])
feat["cat"] = (np.asarray(CAT[h, st, o]) - np.asarray(CAT[h, st, s])).astype(float)
feat["pf"] = np.asarray(Pt[h, o, 12]).astype(float) - np.asarray(Pt[h, s, 12]).astype(float)
for k, v in feat.items(): rows[k] = v
print("rows", len(rows), {g: int(rows.slot.isin(v).sum()) for g, v in groups.items()})
print("gap feature sd", {k: round(float(np.std(v)), 4) for k, v in feat.items()})

def nll(par, r, g, stv, use_b):
    c = np.where(stv == 0, par[0], par[1]); a = expit(c + (par[2] * g if use_b else 0.0))
    return -np.sum(np.log((1 - a) + a * r))

res = {}
for gname, sl in groups.items():
    d = rows[rows.slot.isin(sl)]; r = d.r.values.clip(1e-12, None); stv = d.st.values
    out = {"n": len(d)}
    p0 = minimize(nll, [-0.5, -0.7, 0.0], args=(r, None, stv, False), method="L-BFGS-B"); out["ll0"] = -p0.fun; out["a_pre"], out["a_post"] = expit(p0.x[:2]).tolist()
    for fk in feat:
        g = d[fk].values / (np.std(d[fk].values) + 1e-9)
        p1 = minimize(nll, [p0.x[0], p0.x[1], 0.0], args=(r, g, stv, True), method="L-BFGS-B")
        out[f"{fk}_b"] = float(p1.x[2]); out[f"{fk}_dll"] = float(-p1.fun - out["ll0"])
    # binned activity profile on the hs1 gap (and pf gap): MLE of a within each quintile, by street group
    for fk in ("hs1", "pf"):
        prof = []
        for stg, m_ in (("pre", stv == 0), ("post", stv > 0)):
            dd = d[m_]; qs = np.quantile(dd[fk].values, [0, .2, .4, .6, .8, 1])
            for i in range(5):
                mm = (dd[fk].values >= qs[i]) & (dd[fk].values <= qs[i + 1]); rr = dd.r.values[mm].clip(1e-12, None)
                f = minimize(lambda x: -np.sum(np.log((1 - expit(x[0])) + expit(x[0]) * rr)), [-0.5], method="L-BFGS-B")
                prof.append(dict(street=stg, q=i, lo=float(qs[i]), hi=float(qs[i + 1]), n=int(mm.sum()), a=float(expit(f.x[0])), ll=float(-f.fun)))
        out[f"profile_{fk}"] = prof
    res[gname] = out
    print(gname, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if not k.startswith("profile")}))
    for fk in ("hs1", "pf"):
        print("  profile", fk, " | ".join(f"{p['street']}q{p['q']}[{p['lo']:+.2f},{p['hi']:+.2f}] a={p['a']:.3f} ll={p['ll']:.1f}" for p in out[f"profile_{fk}"]))
json.dump(res, open(f"{R3}/t35_gap_gate.json", "w"), indent=1)
