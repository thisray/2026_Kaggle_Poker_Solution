"""Candidate R2-A (NOT submitted): r11_scoped (LB 0.91479) + eval-only 'pattern-2' (partner-card information dependence) pairs:
risk promoted, predicted_behavior -> other_coordination.  Evidence untouched (paired P+B test of the fourth-family hypothesis)."""
import numpy as np, pandas as pd, hashlib, json, os, sys
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; RAW = f"{A_}/data/raw"
DST = f"{OUT}/r2_candidates"; os.makedirs(DST, exist_ok=True)
TOP_P2, TOP_P2_LOOSE, LOOSE_RANK, DEEP_RANK = 4.0, 3.0, 600, 2000
ZC_THR = 3.02   # s38 combined-score null 1-1e-3 quantile
BASE = os.environ.get("BASE", "r11")
BASEPATH = {"r11": f"{A_}/round11_scoped/r11_scoped.csv", "r15": f"{A_}/round15_campaign/r15_tabicl_blend.csv"}[BASE]
base = pd.read_csv(BASEPATH, dtype=str); base["risk_score"] = base.risk_score.astype(float)
e = pd.read_parquet(f"{OUT}/s23_infoshare_eval.parquet").merge(pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv")[["slot", "pair_id"]], on="slot")
e["p2"] = np.minimum(e.za0 - e.zf0, e.za1 - e.zf1)
b = base.merge(e[["pair_id", "p2", "n_is"]], on="pair_id", how="left"); b["p2"] = b.p2.fillna(0)
b = b.sort_values(["risk_score", "pair_id"], ascending=[False, True]).reset_index(drop=True); b["rk"] = np.arange(1, len(b) + 1)
EXPAND = os.environ.get("EXPAND", "0") == "1"
flag = (b.p2 > TOP_P2) | ((b.p2 > TOP_P2_LOOSE) & (b.rk <= LOOSE_RANK))
if EXPAND:   # transductive f4 expansion (s26): pair-feature 'fourth-family likeness' trained WITHOUT p2, confirmed by moderate p2
    f4 = pd.read_parquet(f"{OUT}/s26_f4_eval.parquet")[["pair_id", "f4"]]; b = b.merge(f4, on="pair_id", how="left"); b["f4"] = b.f4.fillna(0)
    flag = flag | ((b.p2 > 2.5) & (b.f4 > 0.5)) | ((b.p2 > 2.0) & (b.f4 > 0.8))
if os.environ.get("POSTERIOR", "0") == "3":   # r2c set + combined-score (s38: p2 + disjoint later decisions, null 1e-3, 0/280 DT/SP dev hits) in top-1000
    zc = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet")[["pair_id", "zc"]]; b = b.merge(zc, on="pair_id", how="left"); b["zc"] = b.zc.fillna(-9)
    T = ((b.rk <= 600) & (b.p2 > 2.5)) | ((b.rk <= 1000) & (b.zc > ZC_THR))
    D = T & False
elif os.environ.get("POSTERIOR", "0") == "2":   # top-600 only: the later-decision confirmation (s35) rejects all deeper candidates
    T = (b.rk <= 600) & (b.p2 > 2.5)
    D = T & False
elif os.environ.get("POSTERIOR", "0") == "1":   # excess-over-null rules from the rank-bucket scan (docs/30 R2-X1 supplement 4)
    T = (b.rk <= 600) & (b.p2 > 2.5)
    D = ((b.rk > 600) & (b.rk <= 2000) & (b.p2 > 2.5)) | ((b.rk > 2000) & (b.rk <= 5000) & (b.p2 > 3.0)) | ((b.rk > 600) & (b.p2 > 4.0))
else:
    T = flag & (b.rk <= DEEP_RANK)
    D = flag & (b.rk > DEEP_RANK)
new = b.copy()
TOP_AT, DEEP_AT = 250, 485     # insert below these ranks of the remaining (unmoved) ranking
rest = b[~(T | D)].reset_index(drop=True)
def between(k, n):
    hi, lo = rest.risk_score.iloc[k - 1], rest.risk_score.iloc[k]
    assert hi > lo, (k, hi, lo)
    return hi - (hi - lo) * (np.arange(1, n + 1) / (n + 1))
new.loc[T, "risk_score"] = between(TOP_AT, int(T.sum()))
if D.any(): new.loc[D, "risk_score"] = between(DEEP_AT, int(D.sum()))
new.loc[T | D, "predicted_behavior"] = "other_coordination"
cols = ["pair_id", "risk_score", "predicted_behavior"] + [f"evidence_hand_{i}" for i in range(1, 6)]
out = new[cols].set_index("pair_id").loc[base.pair_id].reset_index()
TAGN = ("r2d_p2comb_other" if os.environ.get("POSTERIOR", "0") == "3" else "r2c_p2top600_other" if os.environ.get("POSTERIOR", "0") == "2" else "r2a3_p2post_other" if os.environ.get("POSTERIOR", "0") == "1" else ("r2a2_p2f4_other" if EXPAND else "r2a_p2_other")) + ("" if BASE == "r11" else "_on_r15")
path = f"{DST}/{TAGN}.csv"; out.to_csv(path, index=False)
sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
disk = pd.read_csv(path, dtype=str); raw = pd.read_csv(BASEPATH, dtype=str)
diff_rows = int((disk.to_numpy() != raw.to_numpy()).any(1).sum())
# ---------- validation (paired vs base + legality)
chk = {}
chk["rows"] = len(out); chk["unique_pair_id"] = int(out.pair_id.nunique())
samp = pd.read_csv(f"{RAW}/sample_submission.csv", usecols=["pair_id"]); chk["pair_set_equal_sample"] = bool(set(out.pair_id) == set(samp.pair_id))
chk["risk_in_0_1"] = bool(out.risk_score.between(0, 1).all())
chk["behavior_values"] = out.predicted_behavior.value_counts().to_dict()
EV = [f"evidence_hand_{i}" for i in range(1, 6)]
chk["evidence_identical_to_base"] = bool((out[EV].to_numpy() == base[EV].to_numpy()).all())
chk["rows_changed_risk"] = int((out.risk_score.to_numpy() != base.risk_score.to_numpy()).sum())
chk["rows_changed_behavior"] = int((out.predicted_behavior.to_numpy() != base.predicted_behavior.to_numpy()).sum())
chk["n_top_zone_promoted"] = int(T.sum()); chk["n_deep_promoted"] = int(D.sum())
chk["top_zone_old_ranks"] = b.loc[T, "rk"].tolist(); chk["deep_old_ranks"] = b.loc[D, "rk"].tolist()
chk["top_zone_old_behavior"] = b.loc[T, "predicted_behavior"].value_counts().to_dict()
nr = out.sort_values(["risk_score", "pair_id"], ascending=[False, True]).reset_index(drop=True); nr["rk"] = np.arange(1, len(nr) + 1)
chk["new_rank_range_top_zone"] = [int(nr[nr.pair_id.isin(b.loc[T, "pair_id"])].rk.min()), int(nr[nr.pair_id.isin(b.loc[T, "pair_id"])].rk.max())]
chk["new_rank_range_deep"] = [int(nr[nr.pair_id.isin(b.loc[D, "pair_id"])].rk.min()), int(nr[nr.pair_id.isin(b.loc[D, "pair_id"])].rk.max())] if D.any() else []
chk["sha256"] = sha; chk["on_disk_rows_differing_from_base"] = diff_rows; chk["base"] = BASEPATH
chk["params"] = dict(EXPAND=EXPAND, TOP_P2=TOP_P2, TOP_P2_LOOSE=TOP_P2_LOOSE, LOOSE_RANK=LOOSE_RANK, DEEP_RANK=DEEP_RANK, TOP_AT=TOP_AT, DEEP_AT=DEEP_AT)
json.dump(chk, open(f"{DST}/{TAGN}.receipt.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in chk.items() if k not in ("top_zone_old_ranks", "deep_old_ranks")}, indent=1, default=str))
print("top-zone old ranks:", chk["top_zone_old_ranks"]); print("deep old ranks:", chk["deep_old_ranks"])
