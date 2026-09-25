"""P: mixture likelihood-ratio MIL features from hand-level detector scores (phase-quantile aligned, fold-safe f1)."""
import numpy as np, pandas as pd, time, os, sys
from numba import njit, prange
import aggmod as AG
OUT = os.environ["POKER_WORK_DIR"]; RAW = os.environ["POKER_DATA_DIR"]
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
s0 = np.load(f"{OUT}/m26_handscore_phase0.npy"); sl0 = np.load(f"{OUT}/m26_slot_phase0.npy"); h0 = np.load(f"{OUT}/m26_h_phase0.npy")
s1 = np.load(f"{OUT}/m26_handscore_phase1.npy"); sl1 = np.load(f"{OUT}/m26_slot_phase1.npy"); h1 = np.load(f"{OUT}/m26_h_phase1.npy")
log("loaded", len(s0), len(s1))
# labels / folds / hidden
pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
labels = pd.read_csv(f"{RAW}/development_labels.csv"); ev = pd.read_csv(f"{RAW}/development_evidence.csv")
lo = np.minimum(labels.player_1.map(pmap), labels.player_2.map(pmap)).values; hi = np.maximum(labels.player_1.map(pmap), labels.player_2.map(pmap)).values
labels["slot"] = loc.pool.loc[lo].values * 900 + loc.local.loc[lo].values * 30 + loc.local.loc[hi].values
hidx = pd.read_parquet(f"{OUT}/np/hand_index.parquet"); hmap = dict(zip(hidx.hand_id, hidx.hi))
ev = ev.merge(labels[["pair_id", "slot"]], on="pair_id"); ev["h"] = ev.hand_id.map(hmap)
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); fold_of_pool = dv.groupby("pool").fold.first().reindex(range(400)).values
ref = pd.read_parquet(f"{OUT}/m5_both_train_oof.parquet")
hidden_keys = set(ref[(ref.label == -1) & (ref.oof > 0.3)].key)
pos_slots = set(labels.slot[labels.label == 1]); lab_slot = dict(zip(labels.slot, labels.label))
key_of_slot = {}
# hidden slots via key
lpg = pd.read_parquet(f"{OUT}/player_local_v1.parquet")
mem = np.zeros((400, 30), np.int64)
for pool, g in lpg.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
def slot_key(sl): return mem[sl // 900, (sl % 900) // 30] * 12000 + mem[sl // 900, sl % 30]
# ---- phase-quantile tail probability u
def tail_u(scores, ref_scores):
    srt = np.sort(ref_scores)
    return 1.0 - np.searchsorted(srt, scores, side="left") / len(srt) + 0.5 / len(srt)
keys0 = slot_key(sl0)
neg0 = ~np.isin(sl0, list(pos_slots)) & ~np.isin(keys0, list(hidden_keys))
u0 = tail_u(s0, s0[neg0]); u1 = tail_u(s1, s1)
log("u computed; dev neg hands", int(neg0.sum()))
# ---- f1 histograms on log10(u), fold-specific
bins = np.linspace(-7.5, 0, 31)
def lr_table(u_ev):
    lu = np.log10(np.clip(u_ev, 1e-7, 1))
    c, _ = np.histogram(lu, bins=bins)
    f1 = (c + 0.5) / (c.sum() + 0.5 * len(c))
    f0 = 10 ** bins[1:] - 10 ** bins[:-1]
    return np.clip(f1 / f0, 1e-3, 1e5)
ev_key = pd.Series(np.arange(len(sl0)), index=pd.MultiIndex.from_arrays([sl0, h0]))
ev_idx = ev_key.reindex(pd.MultiIndex.from_arrays([ev.slot.values, ev.h.values])).values
ok = ~np.isnan(ev_idx); ev = ev[ok].copy(); ev["row"] = ev_idx[ok].astype(np.int64); ev["fold"] = fold_of_pool[ev.slot.values // 900]
log("evidence rows matched", len(ev))
FAMS = ["directed_transfer", "soft_play", "coordinated_isolation"]
tabs = {}
for f in list(range(5)) + ["all"]:
    sub = ev if f == "all" else ev[ev.fold != f]
    tabs[f] = {"any": lr_table(u0[sub.row.values])}
    for fm in FAMS: tabs[f][fm] = lr_table(u0[sub[sub.behavior_family == fm].row.values])
RGRID = np.array([0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3])
@njit(parallel=True, cache=True)
def pair_ll(order, starts, lbin, LR, rgrid, out):
    # out[p, k] = sum_h log((1-r_k) + r_k * LR[bin_h]) ; rows sorted by slot
    for p in prange(len(starts) - 1):
        for k in range(len(rgrid)):
            r = rgrid[k]; acc = 0.0
            for j in range(starts[p], starts[p + 1]):
                acc += np.log((1 - r) + r * LR[lbin[order[j]]])
            out[p, k] = acc
def features(u, sl, handmask_rows, fold_of_row, phase):
    rows = np.where(handmask_rows)[0]
    order = rows[np.argsort(sl[rows], kind="stable")]
    ss = sl[order]; starts = np.r_[0, np.flatnonzero(np.diff(ss)) + 1, len(ss)]
    slots = ss[starts[:-1]]
    lbin = np.clip(np.digitize(np.log10(np.clip(u, 1e-7, 1)), bins) - 1, 0, len(bins) - 2)
    res = {"slot": slots, "n_h": np.diff(starts)}
    pf = fold_of_pool[slots // 900] if phase == 0 else None
    for name in ["any"] + FAMS:
        LL = np.zeros((len(slots), len(RGRID)))
        if phase == 0:
            for f in range(5):
                m = pf == f
                idx = np.where(m)[0]
                if len(idx) == 0: continue
                sub_starts = []
                # compute for pairs of fold f with fold-specific table
                tmp = np.zeros((len(slots), len(RGRID)))
                pair_ll(order, starts, lbin, tabs[f][name], RGRID, tmp)
                LL[m] = tmp[m]
        else:
            pair_ll(order, starts, lbin, tabs["all"][name], RGRID, LL)
        res[f"lr_{name}_max"] = LL.max(1); res[f"lr_{name}_rhat"] = RGRID[LL.argmax(1)]
        res[f"lr_{name}_r02"] = LL[:, 2]; res[f"lr_{name}_r05"] = LL[:, 4]
    df = pd.DataFrame(res)
    F = df[[f"lr_{fm}_max" for fm in FAMS]].values
    Fm = F - F.max(1, keepdims=True); P = np.exp(Fm); P = P / P.sum(1, keepdims=True)
    for i, fm in enumerate(FAMS): df[f"lr_post_{fm[:2]}"] = P[:, i]
    df["lr_fam_max"] = F.max(1); df["lr_fam_gap"] = np.sort(F, 1)[:, -1] - np.sort(F, 1)[:, -2]
    return df
for kind, seed, name in [("devsub", 11, "devsub11"), ("devsub", 12, "devsub12"), ("dev", 0, "dev"), ("eval", 0, "eval")]:
    hm = AG.hand_mask(kind, seed)
    if kind == "eval":
        df = features(u1, sl1, hm[h1] == 1, None, 1)
    else:
        df = features(u0, sl0, hm[h0] == 1, None, 0)
    df.to_parquet(f"{OUT}/s3_mil_lr_{name}.parquet"); log(name, df.shape, "pos mean lr_any_max", round(float(df[df.slot.isin(pos_slots)].lr_any_max.mean()), 2), "overall median", round(float(df.lr_any_max.median()), 2))
