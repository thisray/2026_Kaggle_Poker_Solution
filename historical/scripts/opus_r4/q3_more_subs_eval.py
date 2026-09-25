"""R4-Q1c: paired evaluation of pair models trained with extra dev subsamples (r4/m15_<tag>) against their two-subsample twins (shared m15_<twin>), and their effect on the
parameter-free z-fusion of every shared model with dev AP >= FLOOR (the other session's t70 rule). AP_clean on devsub11 / devsub12, pool bootstrap.
Usage: python q3_more_subs_eval.py tag:twin [tag:twin ...]"""
import numpy as np, pandas as pd, glob, os, sys, json
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; FLOOR = 0.96; PAIRS = [a.split(":") for a in sys.argv[1:]]
def ap(y, s):
    o = np.argsort(-s, kind="mergesort"); r = y[o]; tp = np.cumsum(r); k = np.arange(1, len(r) + 1); return float(np.sum(tp / k * r) / max(int(y.sum()), 1))
names = [os.path.basename(f).replace("_train_oof.parquet", "") for f in sorted(glob.glob(f"{O}/m*_train_oof.parquet"))]; names = [n for n in names if os.path.exists(f"{O}/{n}_eval_scores.parquet")]
for src in ("devsub11", "devsub12"):
    base = pd.read_parquet(f"{O}/m15_v6ens_base_train_oof.parquet"); M = base[base.src == src][["key", "pool", "y", "hid"]].set_index("key")
    for n in names:
        d = pd.read_parquet(f"{O}/{n}_train_oof.parquet"); d = d[d.src == src][["key", "oof"]].rename(columns={"oof": n}).set_index("key")
        if len(d) and d.index.is_unique: M = M.join(d, how="left")
    for tag, twin in PAIRS:
        d = pd.read_parquet(f"{O}/r4/m15_{tag}_train_oof.parquet"); d = d[d.src == src][["key", "oof"]].rename(columns={"oof": "NEW_" + tag}).set_index("key"); M = M.join(d, how="left")
    y = M.y.values.astype(int); hid = M.hid.astype(bool).values; msk = ~hid | (y == 1); pools = M.pool.values; up = np.unique(pools); idx_by = {p: np.flatnonzero((pools == p) & msk) for p in up}
    def boot(a, b, n=300, seed=11):
        rs = np.random.default_rng(seed); d_ = []
        for _ in range(n):
            ii = np.concatenate([idx_by[p] for p in rs.choice(up, len(up))]); d_.append(ap(y[ii], a[ii]) - ap(y[ii], b[ii]))
        return float(np.mean(d_)), float(np.mean(np.array(d_) > 0))
    z = lambda c: ((M[c] - M[c].mean()) / M[c].std()).fillna(-5).values
    per = {n: ap(y[msk], M[n].fillna(M[n].min()).values[msk]) for n in names if n in M}; good = [n for n in per if per[n] >= FLOOR]
    if os.environ.get('OLDRECEIPT'): good = [n for n in json.load(open(os.environ['OLDRECEIPT']))['models'] if n in M]   # use exactly the other session's fusion members
    print(f"== {src}: shared models {len(per)}, good {len(good)}")
    for tag, twin in PAIRS:
        a = M["NEW_" + tag].fillna(M["NEW_" + tag].min()).values; b = M["m15_" + twin].fillna(M["m15_" + twin].min()).values; mb, pb = boot(a, b)
        print(f"   single: {tag} {ap(y[msk], a[msk]):.5f} vs twin {twin} {ap(y[msk], b[msk]):.5f}  delta {ap(y[msk], a[msk]) - ap(y[msk], b[msk]):+.5f}  bootstrap mean {mb:+.5f} P>0 {pb:.3f}")
    F0 = np.mean([z(n) for n in good], 0); FA = np.mean([z(n) for n in good] + [z("NEW_" + t) for t, _ in PAIRS], 0)
    FR = np.mean([z(n) for n in good if n not in {"m15_" + tw for _, tw in PAIRS}] + [z("NEW_" + t) for t, _ in PAIRS], 0); m1, p1 = boot(FA, F0); m2, p2 = boot(FR, F0)
    FN = np.mean([z("NEW_" + t) for t, _ in PAIRS], 0); FT = np.mean([z("m15_" + tw) for _, tw in PAIRS], 0); FH = 0.5 * F0 + 0.5 * FN; m3, p3 = boot(FN, F0); m4, p4 = boot(FH, F0); m5, p5 = boot(FN, FT)
    print(f"   NEW-only fusion {ap(y[msk], FN[msk]):.5f} (vs good_z boot {m3:+.5f} P>0 {p3:.3f}; vs fusion of their twins {ap(y[msk], FT[msk]):.5f} boot {m5:+.5f} P>0 {p5:.3f}) | 0.5*good_z + 0.5*NEW {ap(y[msk], FH[msk]):.5f} (boot {m4:+.5f} P>0 {p4:.3f})")
    for wn in (0.3, 0.5, 0.7, 0.85):
        FW = (1 - wn) * F0 + wn * FN; mw, pw = boot(FW, F0); print(f"   grouped fusion WNEW={wn}: {ap(y[msk], FW[msk]):.5f} (vs good_z boot {mw:+.5f} P>0 {pw:.3f})")
    print(f"   fusion good_z {ap(y[msk], F0[msk]):.5f} | + new models {ap(y[msk], FA[msk]):.5f} (boot {m1:+.5f} P>0 {p1:.3f}) | twins replaced by new {ap(y[msk], FR[msk]):.5f} (boot {m2:+.5f} P>0 {p2:.3f})")
