import numpy as np, pandas as pd
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; C = f"{OUT}/r2_candidates"
d = pd.read_parquet(f"{OUT}/s84_pair_bf_dev.parquet"); e = pd.read_parquet(f"{OUT}/s84_pair_bf_eval.parquet")
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); lp = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
dv["slot"] = PI.pair_slot(dv.pool.values, lp.local.loc[dv.p_lo].values, lp.local.loc[dv.p_hi].values)
d = d.merge(dv[["slot", "label", "fam", "oof"]], on="slot", how="left")
for f in ["coordinated_isolation", "directed_transfer", "soft_play"]:
    x = d[(d.label == 1) & (d.fam == f)].bf
    print(f"dev {f[:2]} positives: P(BF>5)={(x > 5).mean():.3f}  P(BF>10)={(x > 10).mean():.3f}  n={len(x)}")
dn = d[d.label != 1]; print("dev U+neg: P(BF>5)=%.5f P(BF>10)=%.6f; dev U top-1000 by oof: P(BF>5)=%.4f" % ((dn.bf > 5).mean(), (dn.bf > 10).mean(), (dn.sort_values('oof', ascending=False).head(1000).bf > 5).mean()))
e = e.merge(pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv"), on="slot"); e["rk"] = e.risk_score.rank(ascending=False)
base = pd.read_csv(f"{C}/r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv", usecols=["pair_id", "predicted_behavior", "risk_score"])
e = e.merge(base.rename(columns={"risk_score": "risk_r2j2m"}), on="pair_id"); e["member"] = e.predicted_behavior == "other_coordination"
e["rk_r2j2m"] = e.risk_r2j2m.rank(ascending=False)
z = pd.read_parquet(f"{OUT}/s77_eval_comb3.parquet")[["slot", "zp", "zq", "zr", "zc3"]]
e = e.merge(z, on="slot", how="left")
print(e[(~e.member) & (e.rk > 600) & (e.bf > 3)].sort_values("bf", ascending=False)[["pair_id", "rk", "rk_r2j2m", "predicted_behavior", "bf", "n_a", "zp", "zq", "zr", "zc3"]].round(2).to_string())
print("top-600 non-members BF>5 by predicted family:", e[(~e.member) & (e.rk <= 600) & (e.bf > 5)].predicted_behavior.value_counts().to_dict())
print("members with BF<0:", e[e.member & (e.bf < 0)][["pair_id", "rk", "bf", "zc3"]].round(2).to_string())
e.drop(columns=[c for c in e.columns if c == "nb"]).to_parquet(f"{OUT}/s85_eval_bf.parquet")
