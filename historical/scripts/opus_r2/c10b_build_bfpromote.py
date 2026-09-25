"""r2j2mB: on the monotone base r2j2m, promote non-member pairs with strong hand-level partner-card evidence
(pair Bayes factor BF>3 from s84, empirical null rate 1.2e-4) that are currently ranked 601-20000, to right after the member
block (ordered by BF), relabelled other_coordination (B-neutral).  Evidence untouched.  Applied to every r2j2m variant."""
import numpy as np, pandas as pd, hashlib, json
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; C = f"{OUT}/r2_candidates"
THR = float(__import__("os").environ.get("BFTHR", "6.0"))
e = pd.read_parquet(f"{OUT}/s85_eval_bf.parquet")
base = pd.read_csv(f"{C}/r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv", dtype=str)
r = base.risk_score.astype(float).values; order0 = base.pair_id.values[np.argsort(-r, kind="stable")]
pos0 = pd.Series(np.arange(len(order0)), index=order0)
mem = set(base[base.predicted_behavior == "other_coordination"].pair_id)
last_member_pos = int(pos0.loc[list(mem)].max())
cand = e[(~e.member) & (e.bf > THR)].copy(); cand["pos0"] = cand.pair_id.map(pos0)
promo = cand[(cand.pos0 >= 600) & (cand.pos0 < 20000)].sort_values("bf", ascending=False)
print(f"last member position {last_member_pos + 1}; promoted {len(promo)} pairs; their current positions (1-based):", sorted((promo.pos0 + 1).astype(int).tolist()))
print(promo[["pair_id", "pos0", "predicted_behavior", "bf", "n_a"]].round(2).to_string())
rest = [p for p in order0 if p not in set(promo.pair_id)]
ins = rest.index(order0[last_member_pos]) + 1
new_order = rest[:ins] + promo.pair_id.tolist() + rest[ins:]
N = len(new_order); risk = pd.Series((N - np.arange(N)) / N, index=new_order)
rec = {"threshold": THR, "promoted": promo.pair_id.tolist(), "files": {}}
import os
FILES = os.environ.get("FILES", "r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv,r2n_NH3_on_r2j2m.csv,r2n_N3_on_r2j2m.csv,r2n_N2_on_r2j2m.csv,r2m_M3_both_on_r2j2m.csv,r2m_M1_clear_raise_on_r2j2m.csv,r2h2ev_on_r2j2m.csv").split(",")
for f in FILES:
    d = pd.read_csv(f"{C}/{f}", dtype=str)
    assert (d.pair_id.values == base.pair_id.values).all() and (d.predicted_behavior.values == base.predicted_behavior.values).all() and (d.risk_score.values == base.risk_score.values).all()
    d["risk_score"] = d.pair_id.map(risk).map(lambda v: repr(float(v)))
    d.loc[d.pair_id.isin(promo.pair_id), "predicted_behavior"] = "other_coordination"
    nm = f.replace("r2j2m", "r2j2mB%d" % int(THR)); d.to_csv(f"{C}/{nm}", index=False)
    rec["files"][nm] = {"sha256": hashlib.sha256(open(f"{C}/{nm}", "rb").read()).hexdigest(), "risk_changed": int((d.risk_score.values != base.risk_score.values).sum()),
                        "behavior_changed": int((d.predicted_behavior.values != base.predicted_behavior.values).sum())}
    print(nm, rec["files"][nm])
json.dump(rec, open(f"{C}/r2j2mB%d_receipt_%s.json" % (int(THR), os.environ.get("RTAG", "main")), "w"), indent=1)
