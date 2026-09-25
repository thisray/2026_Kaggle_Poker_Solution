"""R3-P12: build a candidate whose pair ranking comes from the dev-selected fusion, keeping the deployed structure.

Structure (identical to c4_build_r2j, which produced the LB-verified NDw ranking): non-member pairs sorted by the
fused score; the NDw fourth-family member block (77 pairs) re-inserted right after rank 250; risk = (N - i) / N.
Labels and evidence are taken unchanged from the base candidate (default r13).
Also verifies the reconstruction of the deployed ranking: the same construction with the deployed 3-model rank average
should reproduce the NDw order (reported as Spearman and as the number of pairs whose rank differs).
"""
import numpy as np, pandas as pd, hashlib, json, os
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; C = f"{O}/r2_candidates"
BASE = os.environ.get("BASE", "r13_ndwrank_cinew_f4.csv"); OUTNAME = os.environ["OUTNAME"]; MODS = os.environ.get("MODS", "").split(",")
HOW = os.environ.get("HOW", "z")
loc = pd.read_parquet(f"{O}/player_local_v1.parquet").set_index("player_gi")
sm = pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv")[["slot", "pair_id"]]
def eval_scores(tag):
    t = pd.read_parquet(f"{O}/{tag}_eval_scores.parquet")
    t["slot"] = t.pool * 900 + loc.local.loc[t.p_lo].values * 30 + loc.local.loc[t.p_hi].values
    return t[["slot", "score"]].rename(columns={"score": tag})
E = None
for m in MODS:
    t = eval_scores(m); E = t if E is None else E.merge(t, on="slot", how="inner")
print(f"eval score matrix: {E.shape[0]} pairs x {len(MODS)} models", flush=True)
def fuse(df, mods, how):
    if how == "z": V = [(df[m] - df[m].mean()) / df[m].std() for m in mods]
    elif how == "rank": V = [df[m].rank(pct=True) for m in mods]
    else: V = [np.log(np.clip(df[m], 1e-9, 1 - 1e-9) / (1 - np.clip(df[m], 1e-9, 1 - 1e-9))) for m in mods]
    return np.nanmean(np.stack([np.asarray(v, float) for v in V]), 0)
def group_of(n):
    if "cat" in n: return "cat"
    if any(k in n for k in ("goss", "dart", "extra")): return "lgb_alt"
    if n.startswith("m5"): return "old"
    return "lgb_gbdt"
if os.environ.get("GROUP") == "1":
    G = {}
    for m in MODS: G.setdefault(group_of(m), []).append(m)
    print("groups:", {k: len(v) for k, v in G.items()})
    E["fused"] = np.nanmean(np.stack([fuse(E, v, HOW) for v in G.values()]), 0)
else:
    E["fused"] = fuse(E, MODS, HOW)
s = sm.merge(E[["slot", "fused"]], on="slot")
ndw = pd.read_csv(f"{C}/r2n_NDw_on_r2j2m.csv", dtype=str, keep_default_na=False)
members = set(ndw.pair_id[ndw.predicted_behavior == "other_coordination"])
def build_order(score_col, frame):
    b = frame[["pair_id", score_col]].copy(); b["mem"] = b.pair_id.isin(members)
    rest = b[~b.mem].sort_values([score_col, "pair_id"], ascending=[False, True]); mm = b[b.mem].sort_values([score_col, "pair_id"], ascending=[False, True])
    return pd.concat([rest.iloc[:250], mm, rest.iloc[250:]]).pair_id.values
order = build_order("fused", s)
# reconstruction check of the deployed ranking
DEP = ["m15_v6ens_base", "m15_v6ens_cat", "m15_v6ens_cat11"]
if all(os.path.exists(f"{O}/{d}_eval_scores.parquet") for d in DEP):
    D = None
    for m in DEP:
        t = eval_scores(m); D = t if D is None else D.merge(t, on="slot", how="inner")
    D["dep"] = 0.5 * D[DEP[0]].rank(pct=True) + 0.25 * D[DEP[1]].rank(pct=True) + 0.25 * D[DEP[2]].rank(pct=True)
    sd = sm.merge(D[["slot", "dep"]], on="slot"); dep_order = build_order("dep", sd)
    ndw_order = ndw.sort_values("risk_score", ascending=False, key=lambda c: c.astype(float), kind="mergesort").pair_id.values
    r1 = pd.Series(np.arange(len(dep_order)), index=dep_order); r2 = pd.Series(np.arange(len(ndw_order)), index=ndw_order)
    j = pd.DataFrame({"dep": r1, "ndw": r2.reindex(r1.index)})
    print(f"deployed-ranking reconstruction vs NDw: spearman {j.dep.corr(j.ndw, method='spearman'):.5f}; "
          f"identical positions {int((j.dep == j.ndw).sum())}/{len(j)}; top-600 set overlap "
          f"{len(set(dep_order[:600]) & set(ndw_order[:600]))}/600", flush=True)
base = pd.read_csv(f"{C}/{BASE}", dtype=str, keep_default_na=False)
N = len(order); risk = pd.Series((N - np.arange(N)) / N, index=order)
out = base.copy(); out["risk_score"] = out.pair_id.map(risk).map(lambda v: repr(float(v)))
assert out.risk_score.notna().all()
path = f"{C}/{OUTNAME}.csv"; out.to_csv(path, index=False)
EVC = [f"evidence_hand_{i}" for i in range(1, 6)]
old_rank = base.risk_score.astype(float).rank(ascending=False, method="first"); new_rank = out.risk_score.astype(float).rank(ascending=False, method="first")
rec = dict(file=OUTNAME + ".csv", base=BASE, models=MODS, how=HOW, sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(),
           behavior_identical=bool((out.predicted_behavior.values == base.predicted_behavior.values).all()),
           evidence_identical=bool((out[EVC].values == base[EVC].values).all()),
           rank_changed=int((old_rank != new_rank).sum()), moved_gt_100=int(((new_rank - old_rank).abs() > 100).sum()),
           spearman_vs_base=float(pd.Series(old_rank).corr(pd.Series(new_rank), method="spearman")),
           top600_overlap=int(len(set(base.pair_id[old_rank <= 600]) & set(out.pair_id[new_rank <= 600]))))
json.dump(rec, open(path.replace(".csv", ".receipt.json"), "w"), indent=1); print(json.dumps(rec)[:600])
