"""R4-V2: does adding transductive students to the other session's parameter-free z-fusion improve dev AP_clean?
Members: every m*_train_oof.parquet in the main artifact dir with AP >= FLOOR (same rule as opus_r3/t70), plus r4 students (column `oof`), plus their own bases (`oof_base`) as a control."""
import numpy as np, pandas as pd, glob, os, sys, json
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; FLOOR = float(os.environ.get("FLOOR", "0.96"))
def ap(y, s):
    o = np.argsort(-s, kind="mergesort"); r = y[o]; tp = np.cumsum(r); k = np.arange(1, len(r) + 1); return float(np.sum(tp / k * r) / max(int(y.sum()), 1))
names = [os.path.basename(f).replace("_train_oof.parquet", "") for f in sorted(glob.glob(f"{O}/m*_train_oof.parquet"))]
names = [n for n in names if os.path.exists(f"{O}/{n}_eval_scores.parquet")]
stud = [os.path.basename(f).replace("_train_oof.parquet", "") for f in sorted(glob.glob(f"{O}/r4/m15_*_train_oof.parquet"))]
res = {}
for src in ("devsub11", "devsub12"):
    base = pd.read_parquet(f"{O}/m15_v6ens_base_train_oof.parquet"); M = base[base.src == src][["key", "pool", "y", "hid"]].set_index("key")
    for n in names:
        d = pd.read_parquet(f"{O}/{n}_train_oof.parquet"); d = d[d.src == src][["key", "oof"]].rename(columns={"oof": n}).set_index("key")
        if len(d) and d.index.is_unique: M = M.join(d, how="left")
    for n in stud:
        d = pd.read_parquet(f"{O}/r4/{n}_train_oof.parquet"); d = d[d.src == src].set_index("key")
        M = M.join(d[["oof"]].rename(columns={"oof": "S_" + n}), how="left").join(d[["oof_base"]].rename(columns={"oof_base": "B_" + n}), how="left")
    y = M.y.values.astype(int); hid = M.hid.astype(bool).values; msk = ~hid | (y == 1); pools = M.pool.values
    z = lambda c: ((M[c] - M[c].mean()) / M[c].std()).fillna(-5).values
    per = {n: ap(y[msk], M[n].fillna(M[n].min()).values[msk]) for n in names if n in M}
    good = [n for n in per if per[n] >= FLOOR]
    F0 = np.mean([z(n) for n in good], 0)
    out = {"fusion_good_z": ap(y[msk], F0[msk]), "n_good": len(good)}
    for n in stud:
        out[f"single base {n}"] = ap(y[msk], M["B_" + n].values[msk]); out[f"single student {n}"] = ap(y[msk], M["S_" + n].values[msk])
    FS = np.mean([z(n) for n in good] + [z("S_" + n) for n in stud], 0); FB = np.mean([z(n) for n in good] + [z("B_" + n) for n in stud], 0)
    out["fusion + students"] = ap(y[msk], FS[msk]); out["fusion + their bases (control)"] = ap(y[msk], FB[msk])
    if stud:
        SS = np.mean([z("S_" + n) for n in stud], 0); out["students only"] = ap(y[msk], SS[msk])
        for wt in (0.3, 0.5):
            FW = (1 - wt) * F0 + wt * SS; out[f"fusion*(1-{wt}) + students*{wt}"] = ap(y[msk], FW[msk])
        rs = np.random.default_rng(11); up = np.unique(pools); d_ = []
        idx_by = {p: np.flatnonzero((pools == p) & msk) for p in up}
        for _ in range(300):
            ii = np.concatenate([idx_by[p] for p in rs.choice(up, len(up))]); d_.append(ap(y[ii], FS[ii]) - ap(y[ii], FB[ii]))
        out["bootstrap (fusion+students) - (fusion+bases): mean"] = float(np.mean(d_)); out["P(>0)"] = float(np.mean(np.array(d_) > 0))
    res[src] = out
    print(src); [print(f"   {k:60s} {v:.5f}" if isinstance(v, float) else f"   {k:60s} {v}") for k, v in out.items()]
json.dump(res, open(f"{O}/r4/v2_fusion_with_students.json", "w"), indent=1)
