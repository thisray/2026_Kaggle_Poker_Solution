"""Round-10 P-line: temporal burst / rolling / concentration pair features.

For every (pool, player pair) build sequence features from R_v1 directed interactions
in time order: 8 phase bins, rolling windows, concentration shares, direction runs,
isolation purity. Phase-aware: dev pairs use phase-0 hands, eval pairs phase-1.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

OP = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
OUT = Path(f"{OP}/r10_pburst")
OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

s_player = np.load(f"{OP}/np/s_player.npy", mmap_mode='r')
h_table = np.load(f"{OP}/np/h_table.npy", mmap_mode='r')
h_phase = np.load(f"{OP}/np/h_phase.npy", mmap_mode='r')
h_ts = np.load(f"{OP}/np/h_ts.npy", mmap_mode='r')
h_bb = np.load(f"{OP}/np/h_bb.npy", mmap_mode='r')
R = np.load(f"{OP}/R_v1.npy", mmap_mode='r')
log("arrays loaded")

FEATS = [
    "b_sum_absnet", "b_sum_gross", "b_sum_foldmag", "b_sum_iso", "b_sum_sq", "b_sum_hu", "b_sum_facing",
    "b_top1_share", "b_top3_share", "b_top5_share", "b_hhi",
    "b_roll10_absnet", "b_roll20_absnet", "b_roll10_iso", "b_roll20_iso",
    "b_binmax_absnet", "b_burst_ratio", "b_nbins_hot",
    "b_n_gt1", "b_n_gt2", "b_n_gt5", "b_frac_gt2", "b_rate_absnet", "b_rate_iso",
    "b_pos_argmax", "b_pos_first90", "b_pos5_absnet", "b_max_over_med",
    "b_dir_share", "b_max_run", "b_frac_pos",
    "b_iso_purity", "b_iso_n_gain", "b_fold_gain_share", "b_n_hands", "b_span",
]


def pair_feats(idx, sa, sb):
    order = np.argsort(h_ts[idx], kind='stable')
    idx = idx[order]; sa = sa[order]; sb = sb[order]
    n = len(idx)
    if n == 0:
        return None
    flow_ab = np.asarray(R[idx, sa, sb, 7], float)
    flow_ba = np.asarray(R[idx, sb, sa, 7], float)
    net = flow_ab - flow_ba
    absnet = np.abs(net)
    gross = flow_ab + flow_ba
    foldmag = np.asarray(R[idx, sa, sb, 1], float) + np.asarray(R[idx, sb, sa, 1], float)
    iso = np.asarray(R[idx, sa, sb, 12], float) + np.asarray(R[idx, sb, sa, 12], float)
    sq = np.asarray(R[idx, sa, sb, 11], float) + np.asarray(R[idx, sb, sa, 11], float)
    hu = np.asarray(R[idx, sa, sb, 13], float)
    facing = np.asarray(R[idx, sa, sb, 0], float) + np.asarray(R[idx, sb, sa, 0], float)
    tot = absnet.sum() + 1e-9
    srt = np.sort(absnet)[::-1]
    top1 = srt[0] / tot
    top3 = srt[:3].sum() / tot
    top5 = srt[:5].sum() / tot
    sh = srt / tot
    hhi = float(np.sum(sh ** 2))
    # rolling windows
    def roll_max(x, w):
        if n < w:
            return float(x.sum())
        c = np.concatenate([[0.0], np.cumsum(x)])
        return float((c[w:] - c[:-w]).max())
    # phase bins
    bins = np.minimum((np.arange(n) * 8) // max(n, 1), 7)
    binsum = np.bincount(bins, weights=absnet, minlength=8)
    binmax = float(binsum.max())
    burst = binmax / tot
    thresh = binsum.mean() * 2 + 1e-9
    nbins_hot = float((binsum > thresh).sum())
    # positions
    q90 = np.quantile(absnet, 0.9)
    first90 = float(np.argmax(absnet > q90)) / max(n - 1, 1) if (absnet > q90).any() else 1.0
    posmax = float(np.argmax(absnet)) / max(n - 1, 1)
    order5 = np.argsort(absnet)[::-1][:5]
    pos5 = float(np.sort(order5).max()) / max(n - 1, 1)
    med = np.median(absnet) + 1e-6
    # direction
    dir_share = float(net.sum() / tot)
    signs = np.sign(net)
    run = 0; best_run = 0
    for v in signs:
        if v != 0:
            run = run + 1 if (run == 0 or v == prev) else 1
            prev = v
            best_run = max(best_run, run)
        else:
            run = 0
    frac_pos = float((net > 0).mean())
    # isolation purity
    mi = iso > 0
    iso_purity = float((net[mi] > 0).mean()) if mi.any() else 0.0
    fold_gain = float(foldmag[(net > 0)].sum() / (foldmag.sum() + 1e-9))
    idx_dev = idx
    return [
        absnet.sum(), gross.sum(), foldmag.sum(), iso.sum(), sq.sum(), hu.sum(), facing.sum(),
        top1, top3, top5, hhi,
        roll_max(absnet, 10), roll_max(absnet, 20), roll_max(iso, 10), roll_max(iso, 20),
        binmax, burst, nbins_hot,
        float((absnet > 1).sum()), float((absnet > 2).sum()), float((absnet > 5).sum()),
        float((absnet > 2).mean()), float(absnet.sum() / n), float(iso.sum() / n),
        posmax, first90, pos5, float(srt[0] / med),
        dir_share, float(best_run), frac_pos,
        iso_purity, float((iso[mi] * (net[mi] > 0)).sum()) if mi.any() else 0.0, fold_gain,
        float(n), float(np.ptp(idx_dev)) if n > 1 else 0.0,
    ]


def build(pairs, phase):
    """pairs: DataFrame with pool, p_lo, p_hi (global player gis)."""
    out = np.full((len(pairs), len(FEATS)), np.nan, np.float32)
    pool_arr = pairs.pool.to_numpy()
    lo_arr = pairs.p_lo.to_numpy()
    hi_arr = pairs.p_hi.to_numpy()
    for pool in np.unique(pool_arr):
        sel = np.flatnonzero(pool_arr == pool)
        H = np.flatnonzero((np.asarray(h_table) == pool) & (np.asarray(h_phase) == phase))
        H = H[np.argsort(np.asarray(h_ts)[H], kind='stable')]
        M = np.asarray(s_player[H])  # (nh,6)
        seat_of = {}
        for p in np.unique(M):
            if p < 0:
                continue
            mask = (M == p)
            seat_of[int(p)] = (mask.any(1), mask.argmax(1))
        for i in sel:
            a, b = int(lo_arr[i]), int(hi_arr[i])
            ma, sa_ = seat_of[a]; mb, sb_ = seat_of[b]
            m = ma & mb
            idx = H[m]; sa = sa_[m]; sb = sb_[m]
            fv = pair_feats(idx, sa, sb)
            if fv is not None:
                out[i] = fv
        if pool % 50 == 0:
            log("pool", pool, "phase", phase, "done")
    return out


for split, phase in [("dev", 0), ("eval", 1)]:
    p = pd.read_parquet(f"{OP}/ptab_{split}.parquet", columns=["pool", "p_lo", "p_hi"])
    X = build(p, phase)
    d = pd.DataFrame(X, columns=FEATS)
    d["pool"] = p.pool.to_numpy(); d["p_lo"] = p.p_lo.to_numpy(); d["p_hi"] = p.p_hi.to_numpy()
    d.to_parquet(OUT / f"{split}_burst.parquet")
    log("saved", split, d.shape, "nan", int(np.isnan(X).any(1).sum()))

json.dump({"features": FEATS}, open(OUT / "meta.json", "w"), indent=2)
log("all done")
