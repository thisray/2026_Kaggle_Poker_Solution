"""R3-F4e: the fair test of the pair-win prior.

t93 measured the pw filter against the DEPLOYED per-hand ranking, which already sees who won -- so the filter could
only destroy information. The fourth-family decoder is in a different situation: its per-hand score is a policy
likelihood ratio that knows nothing about the pot. The fair analogue on the known families is a WIN-AGNOSTIC anomaly
score: the largest policy surprisal (-log q, de-memorised: policy_v2 where it did not train, else policy_v1) over the
two members' decisions in that hand. Then compare decoding by score alone vs score with the pair-win prior.
If the prior helps a family whose evidence is only partly pair-wins (CI: 57.6%), keeping it for the fourth family is
justified; if it hurts there too, the fourth-family decoder is throwing away two thirds of its candidates.
"""
import numpy as np, pandas as pd, json
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; O = f"{A_}/opus_r1_20260917"; D = f"{O}/np"
seq = pd.read_parquet(f"{O}/t5_dev_seq.parquet").rename(columns={"sl": "slot"})
won = np.load(f"{D}/s_won.npy", mmap_mode="r"); off = np.load(f"{D}/a_off.npy")
a_seat = np.load(f"{D}/a_seat.npy", mmap_mode="r"); sp = np.load(f"{D}/s_player.npy", mmap_mode="r")
Y = np.load(f"{O}/dec_Y.npy", mmap_mode="r"); P1 = np.load(f"{O}/dec_probs_v1.npy", mmap_mode="r"); P2 = np.load(f"{O}/dec_probs_v2.npy", mmap_mode="r")
N = int(off[-1]); in1 = np.zeros(N, bool); in1[np.random.RandomState(0).choice(N, 4_000_000, replace=False)] = True
in2 = np.zeros(N, bool); in2[np.random.RandomState(1).choice(N, 6_000_000, replace=False)] = True
H, S, T, SL = PI.all_pair_hands(0); m = np.isin(SL, seq.slot.unique()); H, S, T, SL = H[m], S[m], T[m], SL[m]
W = np.asarray(won[H]); ix = np.arange(len(H))
pw = ((W[ix, S] > 0) | (W[ix, T] > 0)).astype(int)
sur_max = np.zeros(len(H)); sur_sum = np.zeros(len(H))
for i, (h, s_, t_) in enumerate(zip(H, S, T)):
    ks = np.arange(off[h], off[h + 1]); seats = np.asarray(a_seat[ks])
    mm = (seats == s_) | (seats == t_)
    km = ks[mm]
    if not len(km): continue
    y = np.asarray(Y[km]); p1 = np.asarray(P1[km])[np.arange(len(km)), y]; p2 = np.asarray(P2[km])[np.arange(len(km)), y]
    q = np.where(~in2[km], p2, np.where(~in1[km], p1, np.nan))
    s = -np.log(np.clip(q, 1e-6, 1.0)); ok = np.isfinite(s)
    if ok.any(): sur_max[i] = float(np.nanmax(s[ok])); sur_sum[i] = float(np.nansum(s[ok]))
    if i % 10000 == 0: print(f"  {i}/{len(H)}", flush=True)
g = pd.DataFrame({"slot": SL, "h": H, "pw": pw, "sur_max": sur_max, "sur_sum": sur_sum})
f = seq[["slot", "h", "fam", "ev"]].merge(g, on=["slot", "h"], how="inner")
den = seq.groupby("slot").ev.sum().to_dict()
def ap5(x, col):
    top = x.sort_values(col, ascending=False, kind="mergesort").head(5).ev.values
    hits = 0; s = 0.0
    for i, e in enumerate(top):
        if e: hits += 1; s += hits / (i + 1)
    return s / min(5, int(den[x.name]))
res = {}
for fam, x in f.groupby("fam"):
    x = x.copy()
    for base in ("sur_max", "sur_sum"):
        x[f"{base}_hardpw"] = x[base] - 1e6 * (1 - x.pw)                       # forbid non-pair-wins (the F4 decoder)
        x[f"{base}_softpw"] = x[base] * (0.35 + 0.65 * x.pw)                   # soft prior
    cols = [c for c in x.columns if c.startswith("sur_")]
    e = {c: float(x.groupby("slot").apply(lambda z: ap5(z, c), include_groups=False).mean()) for c in cols}
    res[fam] = {k: round(v, 4) for k, v in e.items()}; res[fam]["pairs"] = int(x.slot.nunique())
    res[fam]["gain_hard_over_plain"] = round(e["sur_max_hardpw"] - e["sur_max"], 4)
    print(fam, json.dumps(res[fam]), flush=True)
json.dump(res, open(f"{O}/r3/t94_pw_prior_fair.json", "w"), indent=1)
