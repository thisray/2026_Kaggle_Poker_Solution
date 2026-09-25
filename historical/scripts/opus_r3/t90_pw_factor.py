"""R3-F4a: is the "the pair won this hand" factor in the fourth-family decoder justified?

`t37_re2d.py` multiplies every per-hand event probability by pw = (either member won the pot) before the first-5 DP.
That factor was never tested. The known families are the only place it can be tested, and the fourth family comes out
of the same generator, so the transfer is principled: measure, per known family, how much more likely a LISTED evidence
hand is to be a pair-win than a co-seated non-evidence hand of the same pair.
Also checks whether listed evidence is chronological (the assumption behind the first-5 DP).
"""
import numpy as np, pandas as pd, json
import pairindex as PI
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D = f"{O}/np"
seq = pd.read_parquet(f"{O}/t5_dev_seq.parquet")
seq = seq.rename(columns={c: "slot" for c in seq.columns if c in ("sl",)})
print("t5_dev_seq columns:", list(seq.columns)[:12], "rows", len(seq))
won = np.load(f"{D}/s_won.npy", mmap_mode="r"); ts = np.load(f"{D}/h_ts.npy")
H, S, T, SL = PI.all_pair_hands(0)
m = np.isin(SL, seq.slot.unique()); H, S, T, SL = H[m], S[m], T[m], SL[m]
W = np.asarray(won[H]); ix = np.arange(len(H))
pw = pd.DataFrame({"slot": SL, "h": H, "ts": ts[H],
                   "pw": ((W[ix, S] > 0) | (W[ix, T] > 0)).astype(int),
                   "w_s": (W[ix, S] > 0).astype(int), "w_t": (W[ix, T] > 0).astype(int)})
f = seq[["slot", "h", "fam", "ev"]].merge(pw, on=["slot", "h"], how="inner")
print("merged rows", len(f), "evidence rows", int(f.ev.sum()))
out = {}
for fam, g in f.groupby("fam"):
    ev1 = g[g.ev == 1]; ev0 = g[g.ev == 0]
    r = dict(pairs=int(g.slot.nunique()), evidence=len(ev1), other=len(ev0),
             pw_evidence=round(float(ev1.pw.mean()), 4), pw_other=round(float(ev0.pw.mean()), 4))
    r["lift"] = round(r["pw_evidence"] / max(r["pw_other"], 1e-9), 3)
    # within-pair version (removes pair-level differences in win rate)
    d = g.groupby("slot").apply(lambda x: pd.Series(dict(a=x.pw[x.ev == 1].mean(), b=x.pw[x.ev == 0].mean())), include_groups=False).dropna()
    r["within_pair_delta"] = round(float((d.a - d.b).mean()), 4); r["within_pair_pairs"] = len(d)
    # chronology: is the evidence set the FIRST k pair-win hands / first k hands?
    def rank_of_evidence(x):
        x = x.sort_values("ts"); pos = np.flatnonzero(x.ev.values == 1)
        return pd.Series(dict(n=len(x), k=len(pos), max_rank=(pos.max() + 1) if len(pos) else np.nan,
                              first_k=int(len(pos) > 0 and pos.max() + 1 == len(pos))))
    c = g.groupby("slot").apply(rank_of_evidence, include_groups=False)
    r["share_evidence_is_the_first_k_hands"] = round(float(c.first_k.mean()), 4)
    cw = g[g.pw == 1].groupby("slot").apply(rank_of_evidence, include_groups=False)
    r["share_evidence_is_the_first_k_pairwin_hands"] = round(float(cw.first_k.mean()), 4)
    out[fam] = r
    print(fam, json.dumps(r), flush=True)
json.dump(out, open(f"{O}/r3/t90_pw_factor.json", "w"), indent=1)
