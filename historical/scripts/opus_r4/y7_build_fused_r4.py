"""R4-Y7: candidate whose pair ranking fuses two GROUPS of pair models on the z scale: OLD = shared two-subsample models (the other session's t70 'good' rule: dev AP_clean >= FLOOR),
NEW = R4 models trained with extra exposure-matched dev subsamples (r4/m15_<tag>). fused = (1 - WNEW) * mean_z(OLD) + WNEW * mean_z(NEW).
Structure identical to opus_r3/t71_build_fused_rank.py (LB-verified NDw construction): non-member pairs sorted by the fused score, the 77 NDw fourth-family members re-inserted
after rank 250, risk = (N - i) / N. Labels and evidence are copied unchanged from BASE (in r4/cand or r2_candidates).
Usage: WNEW=0.5 python y7_build_fused_r4.py <base csv path> <out name> <new_tag,new_tag,...>"""
import numpy as np, pandas as pd, hashlib, json, os, sys, glob
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; C = f"{O}/r2_candidates"
BASE, OUTNAME, NEW = sys.argv[1], sys.argv[2], sys.argv[3].split(","); WNEW = float(os.environ.get("WNEW", 0.5)); FLOOR = float(os.environ.get("FLOOR", 0.96))
def ap(y, s):
    o = np.argsort(-s, kind="mergesort"); r = y[o]; tp = np.cumsum(r); k = np.arange(1, len(r) + 1); return float(np.sum(tp / k * r) / max(int(y.sum()), 1))
names = [os.path.basename(f).replace("_train_oof.parquet", "") for f in sorted(glob.glob(f"{O}/m*_train_oof.parquet"))]; names = [n for n in names if os.path.exists(f"{O}/{n}_eval_scores.parquet")]
base_oof = pd.read_parquet(f"{O}/m15_v6ens_base_train_oof.parquet"); per = {}
for n in names:
    d = pd.read_parquet(f"{O}/{n}_train_oof.parquet"); v = []
    for src in ("devsub11", "devsub12"):
        b = base_oof[base_oof.src == src][["key", "y", "hid"]].set_index("key"); x = d[d.src == src][["key", "oof"]].set_index("key")
        if not len(x) or not x.index.is_unique: v = []; break
        j = b.join(x, how="left"); msk = (~j.hid.astype(bool) | (j.y == 1)).values; v.append(ap(j.y.values.astype(int)[msk], j.oof.fillna(j.oof.min()).values[msk]))
    if v: per[n] = float(np.mean(v))
OLD = sorted(n for n, a in per.items() if a >= FLOOR)
if os.environ.get("OLDRECEIPT"): OLD = [n for n in json.load(open(os.environ["OLDRECEIPT"]))["models"] if os.path.exists(f"{O}/{n}_eval_scores.parquet")]   # exactly the other session's fusion members
print(f"OLD group: {len(OLD)} models (floor {FLOOR}); NEW group: {NEW}; WNEW {WNEW}", flush=True)
loc = pd.read_parquet(f"{O}/player_local_v1.parquet").set_index("player_gi"); sm = pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv")[["slot", "pair_id"]]
def eval_z(path, tag):
    t = pd.read_parquet(path); t["slot"] = t.pool * 900 + loc.local.loc[t.p_lo].values * 30 + loc.local.loc[t.p_hi].values; t = t.set_index("slot").score; return ((t - t.mean()) / t.std()).rename(tag)
ZO = pd.concat([eval_z(f"{O}/{n}_eval_scores.parquet", n) for n in OLD], axis=1, join="inner"); ZN = pd.concat([eval_z(f"{O}/r4/m15_{n}_eval_scores.parquet", n) for n in NEW], axis=1, join="inner")
F = pd.DataFrame({"fused": (1 - WNEW) * ZO.mean(axis=1) + WNEW * ZN.mean(axis=1).reindex(ZO.index)}).dropna().reset_index(); s = sm.merge(F, on="slot"); assert len(s) == len(sm), "eval coverage"
ndw = pd.read_csv(f"{C}/r2n_NDw_on_r2j2m.csv", dtype=str, keep_default_na=False); members = set(ndw.pair_id[ndw.predicted_behavior == "other_coordination"])
b_ = s[["pair_id", "fused"]].copy(); b_["mem"] = b_.pair_id.isin(members)
rest = b_[~b_.mem].sort_values(["fused", "pair_id"], ascending=[False, True]); mm = b_[b_.mem].sort_values(["fused", "pair_id"], ascending=[False, True]); order = pd.concat([rest.iloc[:250], mm, rest.iloc[250:]]).pair_id.values
base = pd.read_csv(BASE, dtype=str, keep_default_na=False); N = len(order); risk = pd.Series((N - np.arange(N)) / N, index=order)
out = base.copy(); out["risk_score"] = out.pair_id.map(risk).map(lambda v: repr(float(v))); assert out.risk_score.notna().all()
os.makedirs(f"{O}/r4/cand", exist_ok=True); path = f"{O}/r4/cand/{OUTNAME}.csv"; out.to_csv(path, index=False)
EVC = [f"evidence_hand_{i}" for i in range(1, 6)]; old_rank = base.risk_score.astype(float).rank(ascending=False, method="first"); new_rank = out.risk_score.astype(float).rank(ascending=False, method="first")
rec = dict(file=OUTNAME + ".csv", base=os.path.basename(BASE), old_models=OLD, new_models=NEW, wnew=WNEW, sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(),
           behavior_identical=bool((out.predicted_behavior.values == base.predicted_behavior.values).all()), evidence_identical=bool((out[EVC].values == base[EVC].values).all()),
           spearman_vs_base=float(pd.Series(old_rank).corr(pd.Series(new_rank), method="spearman")), top450_overlap=int(len(set(base.pair_id[old_rank <= 450]) & set(out.pair_id[new_rank <= 450]))),
           top600_overlap=int(len(set(base.pair_id[old_rank <= 600]) & set(out.pair_id[new_rank <= 600]))))
json.dump(rec, open(path.replace(".csv", ".receipt.json"), "w"), indent=1); print(json.dumps({k: v for k, v in rec.items() if k != "old_models"}))
