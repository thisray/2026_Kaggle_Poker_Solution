"""R3-P11: choose a pair-ranking fusion on dev and write the eval ranking.

Every pair model with both a dev OOF and an eval score file is a candidate member. Fusions are parameter-free
(equal-weight mean of z-scores / rank percentiles / logits) over a member set chosen by a dev-AP floor. Selection uses
the MEAN of the two exposure-matched subsamples (devsub11, devsub12) so a fusion has to work on both, and a pool
bootstrap reports the probability that it beats the deployed three-model rank average.
With WRITE=1 the winning fusion is applied to the eval score files and the new pair order is saved.
"""
import numpy as np, pandas as pd, glob, os, json
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
FLOOR = float(os.environ.get("FLOOR", "0.96")); WRITE = os.environ.get("WRITE") == "1"
def ap(y, s):
    o = np.argsort(-s, kind="mergesort"); r = y[o]; tp = np.cumsum(r); k = np.arange(1, len(r) + 1)
    return float(np.sum(tp / k * r) / max(int(y.sum()), 1))
files = sorted(glob.glob(f"{O}/m*_train_oof.parquet"))
names = [os.path.basename(f).replace("_train_oof.parquet", "") for f in files]
names = [n for n in names if os.path.exists(f"{O}/{n}_eval_scores.parquet")]
print(f"models with dev OOF and eval scores: {len(names)}", flush=True)
K = {}; Y = {}; POOL = {}; HID = {}
for src in ("devsub11", "devsub12"):
    base = pd.read_parquet(f"{O}/m15_v6ens_base_train_oof.parquet"); b = base[base.src == src][["key", "pool", "y", "hid"]].set_index("key")
    M = b.copy()
    for n in names:
        d = pd.read_parquet(f"{O}/{n}_train_oof.parquet"); d = d[d.src == src][["key", "oof"]].rename(columns={"oof": n}).set_index("key")
        if len(d) and d.index.is_unique: M = M.join(d, how="left")
    K[src] = M; Y[src] = M.y.values.astype(int); POOL[src] = M.pool.values; HID[src] = M.hid.astype(bool).values
avail = [n for n in names if all(n in K[s] and K[s][n].notna().mean() > 0.9 for s in K)]
def ap_clean(s, v): m = ~HID[s] | (Y[s] == 1); return ap(Y[s][m], v[m])
def ap_hidpos(s, v): yy = np.where(HID[s], 1, Y[s]); return ap(yy, v)
per = {n: {s: ap_clean(s, K[s][n].fillna(K[s][n].min()).values) for s in K} for n in avail}
for n, v in sorted(per.items(), key=lambda kv: -np.mean(list(kv[1].values()))):
    print(f"   {n:26s} devsub11 {v['devsub11']:.5f} devsub12 {v['devsub12']:.5f} mean {np.mean(list(v.values())):.5f}")
good = [n for n in avail if np.mean(list(per[n].values())) >= FLOOR]
DEP = [c for c in ("m15_v6ens_base", "m15_v6ens_cat", "m15_v6ens_cat11") if c in avail]
def fuse(M, mods, how):
    if how == "z": V = [(M[m] - M[m].mean()) / M[m].std() for m in mods]
    elif how == "rank": V = [M[m].rank(pct=True) for m in mods]
    else: V = [np.log(np.clip(M[m], 1e-9, 1 - 1e-9) / (1 - np.clip(M[m], 1e-9, 1 - 1e-9))) for m in mods]
    return np.nanmean(np.stack([np.asarray(v, float) for v in V]), 0)
cands = {"deployed": lambda M: 0.5 * M[DEP[0]].rank(pct=True).values + sum(0.5 / max(len(DEP) - 1, 1) * M[c].rank(pct=True).values for c in DEP[1:])}
for how in ("z", "rank", "logit"):
    cands[f"all_{how}"] = (lambda mods, how=how: (lambda M: fuse(M, mods, how)))(avail)
    cands[f"good_{how}"] = (lambda mods, how=how: (lambda M: fuse(M, mods, how)))(good)
def group_of(n):
    if "cat" in n: return "cat"
    if any(k in n for k in ("goss", "dart", "extra")): return "lgb_alt"
    if n.startswith("m5") or "mlp" in n: return "old"
    return "lgb_gbdt"
GRP = {}
for n in avail: GRP.setdefault(group_of(n), []).append(n)
print("groups:", {k: len(v) for k, v in GRP.items()})
def group_fuse(M, how):
    return np.nanmean(np.stack([fuse(M, v, how) for v in GRP.values()]), 0)
for how in ("z", "logit"):
    cands[f"grp_{how}"] = (lambda how=how: (lambda M: group_fuse(M, how)))()
GRP_NOMLP = {k: [m for m in v if "mlp" not in m] for k, v in GRP.items()}
cands["grp_logit_nomlp"] = lambda M: np.nanmean(np.stack([fuse(M, v, "logit") for v in GRP_NOMLP.values() if v]), 0)
GRP_ML = {}
for n in avail: GRP_ML.setdefault("mlp" if "mlp" in n else group_of(n), []).append(n)
cands["grp_logit_mlpgroup"] = lambda M: np.nanmean(np.stack([fuse(M, v, "logit") for v in GRP_ML.values()]), 0)
FAM = [m for m in avail if "o_fam_" in m]
if FAM:
    print("family-specialised models:", FAM)
    def zc(M, m): return ((M[m] - M[m].mean()) / M[m].std()).values
    def fam_max(M):
        byfam = {}
        for m in FAM: byfam.setdefault(m.split("o_fam_")[1].rstrip("0123456789"), []).append(m)
        return np.nanmax(np.stack([np.nanmean(np.stack([zc(M, m) for m in v]), 0) for v in byfam.values()]), 0)
    core = [m for m in avail if m not in FAM]
    cands["core_z_noFam"] = lambda M: fuse(M, core, "z")
    cands["core_z_plus_fammax"] = lambda M: 0.5 * ((lambda v: (v - v.mean()) / v.std())(fuse(M, core, "z")) ) + 0.5 * ((lambda v: (v - v.mean()) / v.std())(fam_max(M)))
    cands["fammax_only"] = lambda M: fam_max(M)
scores = {}
for nm, fn in cands.items():
    a = {s: ap_clean(s, fn(K[s])) for s in K}; h = {s + "_hidpos": ap_hidpos(s, fn(K[s])) for s in K}
    scores[nm] = dict(a, **h, mean=float(np.mean(list(a.values()))), mean_hidpos=float(np.mean(list(h.values()))))
print("\nfusions (dev AP):")
for nm, v in sorted(scores.items(), key=lambda kv: -kv[1]["mean"]):
    print(f"   {nm:14s} clean {v['mean']:.5f} (d11 {v['devsub11']:.5f} d12 {v['devsub12']:.5f}) | hidden-as-positive {v['mean_hidpos']:.5f} | delta_clean {v['mean'] - scores['deployed'][ 'mean']:+.5f} delta_hidpos {v['mean_hidpos'] - scores['deployed'][ 'mean_hidpos']:+.5f}")
best = max((k for k in scores if k != "deployed"), key=lambda k: scores[k]["mean"])
for s in K:
    d = []; pools = POOL[s]; up = np.unique(pools); rs = np.random.default_rng(11)
    bs = cands[best](K[s]); dp = cands["deployed"](K[s]); msk = ~HID[s] | (Y[s] == 1)
    for _ in range(200):
        pick = rs.choice(up, len(up), replace=True); idx = np.concatenate([np.flatnonzero((pools == p) & msk) for p in pick])
        d.append(ap(Y[s][idx], bs[idx]) - ap(Y[s][idx], dp[idx]))
    print(f"   bootstrap {s}: mean {np.mean(d):+.5f} P(>0) {np.mean(np.array(d) > 0):.3f}")
json.dump(dict(per_model={k: v for k, v in per.items()}, fusions=scores, best=best, good=good, deployed=DEP),
          open(f"{O}/r3/t70_fusion_select.json", "w"), indent=1)
print("\nbest fusion:", best, "members:", len(good if best.startswith('good') else avail))
if WRITE:
    mods = good if best.startswith("good") else avail
    how = best.split("_")[1]
    loc = pd.read_parquet(f"{O}/player_local_v1.parquet").set_index("player_gi")
    E = None
    for n in mods:
        t = pd.read_parquet(f"{O}/{n}_eval_scores.parquet")
        t["slot"] = t.pool * 900 + loc.local.loc[t.p_lo].values * 30 + loc.local.loc[t.p_hi].values
        t = t[["slot", "score"]].rename(columns={"score": n}).set_index("slot")
        E = t if E is None else E.join(t, how="inner")
    print("eval fusion rows", len(E), "models", len(mods))
    E["fused"] = fuse(E, mods, how)
    E.reset_index()[["slot", "fused"]].to_parquet(f"{O}/r3/t70_eval_fused.parquet")
    print("wrote", f"{O}/r3/t70_eval_fused.parquet")
