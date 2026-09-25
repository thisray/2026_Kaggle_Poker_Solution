"""Monotone insertion of the 77 fourth-family members (fix suggested by the Round16 audit): members whose position in the
un-moved LGB+Cat ranking is already better than the insertion point keep that position; the others are inserted right
after the 250th non-member (block order by mix).  Risk is then (N - position)/N.  Applied to every r2j2-based evidence
variant (only the risk column changes; behaviour/evidence untouched)."""
import numpy as np, pandas as pd, hashlib, json
from scipy.stats import rankdata
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; C = f"{OUT}/r2_candidates"
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
def load(tag):
    t = pd.read_parquet(f"{OUT}/m15_{tag}_eval_scores.parquet"); t["slot"] = t.pool * 900 + loc.local.loc[t.p_lo].values * 30 + loc.local.loc[t.p_hi].values; return t[["slot", "score"]]
sm = pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv")[["slot", "pair_id"]]
s = sm.merge(load("v6ens_base").rename(columns={"score": "lgb"}), on="slot")
for i, t in enumerate(["v6ens_cat", "v6ens_cat11"]): s = s.merge(load(t).rename(columns={"score": f"cat{i}"}), on="slot")
s["mix"] = 0.5 * rankdata(s.lgb) / len(s) + sum((1 - 0.5) / 2 * rankdata(s[f"cat{i}"]) / len(s) for i in range(2))   # exactly as c4 (same float association)
base = pd.read_csv(f"{C}/r2j2_lgbcat2_p2comb_other_ev_on_r15.csv", dtype=str)
b = base[["pair_id", "predicted_behavior"]].merge(s[["pair_id", "mix"]], on="pair_id"); assert len(b) == len(base)
mem = (b.predicted_behavior == "other_coordination").values
allo = b.sort_values(["mix", "pair_id"], ascending=[False, True]).reset_index(drop=True); allo["brank"] = np.arange(len(allo))   # 0-based base rank
nonm = allo[allo.predicted_behavior != "other_coordination"]
R250 = nonm.brank.iloc[249]                      # base rank of the 250th non-member
mm = allo[allo.predicted_behavior == "other_coordination"].copy()
mm["key"] = np.where(mm.brank < R250, mm.brank.astype(float), R250 + 0.5 + 1e-4 * np.arange(len(mm)))
nonm = nonm.assign(key=nonm.brank.astype(float))
order = pd.concat([nonm, mm]).sort_values("key").pair_id.values
N = len(order); risk = pd.Series((N - np.arange(N)) / N, index=order)
# reproduce the r2j2 block order to report movement
blk = pd.concat([nonm.iloc[:250], mm.sort_values("brank"), nonm.iloc[250:]]).pair_id.values
posb = pd.Series(np.arange(N), index=blk); posm = pd.Series(np.arange(N), index=order)
mids = mm.pair_id.values
kept = mm[mm.brank < R250]
print(f"R250 (base rank of 250th non-member) = {R250}; members kept above the block: {len(kept)} (base ranks {sorted((kept.brank + 1).tolist())})")
print("member positions block->monotone (1-based) for kept members:", [(int(posb[p]) + 1, int(posm[p]) + 1) for p in kept.pair_id])
chk_r2j2 = np.allclose(pd.Series(base.risk_score.astype(float).values, index=base.pair_id).loc[blk].values, (N - np.arange(N)) / N)
print("block order reproduces r2j2 risk exactly:", chk_r2j2); assert chk_r2j2
rec = {}
for f in ["r2j2_lgbcat2_p2comb_other_ev_on_r15.csv", "r2n_N3_on_r2j2.csv", "r2n_N2_on_r2j2.csv", "r2n_N1b_on_r2j2.csv", "r2m_M1_clear_raise_on_r2j2.csv", "r2m_M3_both_on_r2j2.csv", "r2h2ev_on_r2j2.csv"]:
    d = pd.read_csv(f"{C}/{f}", dtype=str)
    assert (d.predicted_behavior.values == base.predicted_behavior.values).all() and (d.pair_id.values == base.pair_id.values).all()
    d["risk_score"] = d.pair_id.map(risk).map(lambda v: repr(float(v)))
    nm = f.replace("_on_r2j2.csv", "_on_r2j2m.csv").replace("r2j2_lgbcat2_p2comb_other_ev_on_r15.csv", "r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv")
    d.to_csv(f"{C}/{nm}", index=False)
    rec[nm] = {"from": f, "sha256": hashlib.sha256(open(f"{C}/{nm}", "rb").read()).hexdigest(), "risk_changed_rows": int((d.risk_score.values != base.risk_score.values).sum())}
    print(nm, rec[nm])
json.dump({"R250": int(R250), "kept": kept.pair_id.tolist(), "files": rec}, open(f"{C}/r2j2m_receipt.json", "w"), indent=1)
