"""R4-X11: per-hand directed interaction kernels (R_v1: 26 channels for the ordered member pair, both directions) and per-player hand summaries (P_v1: 20 channels) for every
co-seated hand, oriented by the role parquet's receiver/sender seats. These are the raw inputs of the R1-R11 upstream hand scores (omniscient-equity conditioned
fold/call/check features), available for all hands of both phases; the censored-event models did not have them.
Usage: python x11_kernel_feats.py <role parquet with slot,h,rs,ss> <out parquet>"""
import numpy as np, pandas as pd, sys
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
d = pd.read_parquet(sys.argv[1])[["slot", "h", "rs", "ss"]].reset_index(drop=True)
lines = open(f"{OUT}/feature_names_v1.txt").read().splitlines(); rn = lines[0][2:].split(","); pn = lines[1][2:].split(",")
R = np.load(f"{OUT}/R_v1.npy", mmap_mode="r"); P = np.load(f"{OUT}/P_v1.npy", mmap_mode="r")
order = np.argsort(d.h.values, kind="stable"); h = d.h.values[order]; rs = d.rs.values[order].astype(int); ss = d.ss.values[order].astype(int)
out = np.zeros((len(d), 2 * len(rn) + 2 * len(pn)), np.float32); CH = 20000
for i in range(0, len(d), CH):
    hh = h[i:i + CH]; a = rs[i:i + CH]; b = ss[i:i + CH]; Rb = np.asarray(R[hh]); Pb = np.asarray(P[hh]); ix = np.arange(len(hh))
    out[order[i:i + CH]] = np.concatenate([Rb[ix, a, b], Rb[ix, b, a], Pb[ix, a], Pb[ix, b]], axis=1)
    print(f"  {i}/{len(d)}", flush=True)
cols = [f"k_rs_{c}" for c in rn] + [f"k_sr_{c}" for c in rn] + [f"k_pr_{c}" for c in pn] + [f"k_ps_{c}" for c in pn]
res = pd.concat([d[["slot", "h"]], pd.DataFrame(out, columns=cols)], axis=1); res.to_parquet(sys.argv[2]); print("saved", sys.argv[2], res.shape)
