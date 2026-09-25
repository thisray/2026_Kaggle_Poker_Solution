"""R3 quick checks (2026-09-19 evening).
A. Does the bet/raise SIZE carry own-card information given the street/facing context? (R19 proposes P(size | state, cards)
   as a new partner-card channel; if normal sizes do not depend on cards, the channel is empty.)
B. Member table of the current other_coordination pairs: substitution BF, placebo D, pair planting rate, family-model
   probabilities, rank - to find members that look like known-family tails rather than partner-card substitution."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"; R3 = f"{OUT}/r3"
L = lambda n: np.load(f"{D}/{n}.npy", mmap_mode="r")
off = np.load(f"{D}/a_off.npy"); a_st = L("a_st"); a_seat = L("a_seat"); a_act = L("a_act"); a_amt = L("a_amount"); a_tc = L("a_to_call"); a_pot = L("a_pot_before"); a_stk = L("a_stack_before")
HS1 = np.load(f"{OUT}/HS1.npy", mmap_mode="r"); Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r")
rng = np.random.default_rng(3); N = len(a_act); K = np.sort(rng.choice(N, 3_000_000, replace=False))
act = np.asarray(a_act[K]); amt = np.asarray(a_amt[K]).astype(float); tc = np.asarray(a_tc[K]).astype(float); pot = np.asarray(a_pot[K]).astype(float)
stk = np.asarray(a_stk[K]).astype(float); st = np.asarray(a_st[K]); seat = np.asarray(a_seat[K])
aggr = (act == 3) | (act == 4) | ((act == 5) & (amt > tc)); allin = act == 5
H = np.searchsorted(off, K, side="right") - 1
m = aggr & (pot > 0); K2, H2, st2, seat2 = K[m], H[m], st[m], seat[m]
size = (amt[m] - tc[m]) / pot[m]; facing = tc[m] > 0; ai = allin[m]
hs1 = np.asarray(HS1[H2, st2, seat2]).astype(float); pf = np.asarray(Pt[H2, seat2, 12]).astype(float)
print(f"A. aggressive decisions sampled {m.sum()} of {len(K)}; all-in share {ai.mean():.3f}")
for sname, sm in (("pre", st2 == 0), ("post", st2 > 0)):
    for fname, fm in (("open", ~facing), ("facing", facing)):
        mm = sm & fm & ~ai; x = size[mm]; y = hs1[mm] if sname == "post" else pf[mm]
        if mm.sum() < 1000: continue
        rho_s = spearmanr(x, y).correlation; u = np.unique(np.round(x, 2)); qs = np.quantile(x, [0, .2, .4, .6, .8, 1])
        prof = [float(y[(x >= qs[i]) & (x <= qs[i + 1])].mean()) for i in range(5)]
        top = pd.Series(np.round(x, 2)).value_counts(normalize=True).head(5)
        print(f"  {sname:4s} {fname:6s} n={mm.sum():7d} spearman(size, strength)={rho_s:+.3f} unique_sizes={len(u)} strength by size quintile={np.round(prof, 3).tolist()} top sizes={dict(top.round(3))}")
        ai_m = sm & fm; print(f"        all-in rate by strength quintile: {[round(float(ai[ai_m][(v >= a) & (v <= b)].mean()), 3) for v in [(hs1 if sname == 'post' else pf)[ai_m]] for a, b in zip(np.quantile(v, [0, .2, .4, .6, .8]), np.quantile(v, [.2, .4, .6, .8, 1]))]}")

print("B. member table")
base = pd.read_csv(f"{C}/r5_subh.csv", dtype=str); base["rk"] = base.risk_score.astype(float).rank(ascending=False, method="first")
oth = base[base.predicted_behavior == "other_coordination"][["pair_id", "rk"]]
bf = pd.read_parquet(f"{R3}/t17a_bf_eval.parquet")[["pair_id", "slot", "bf_sub", "bf_hier", "bf", "nh"]]
pl = pd.concat([pd.read_parquet(f"{R3}/t26_placebo.parquet"), pd.read_parquet(f"{R3}/t27_placebo_ext.parquet")]).drop_duplicates("pair_id")[["pair_id", "D", "n", "set"]]
re_ = pd.read_parquet(f"{R3}/t34_re_pairs.parquet")[["pair_id", "rho_mean", "hands"]]
m7 = pd.read_parquet(f"{OUT}/m7_family_eval.parquet"); loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
lo = m7.key // 12000; hi = m7.key % 12000; m7["slot"] = loc.pool.loc[lo].values * 900 + loc.local.loc[lo].values * 30 + loc.local.loc[hi].values
t = oth.merge(bf, on="pair_id", how="left").merge(pl, on="pair_id", how="left").merge(re_, on="pair_id", how="left").merge(m7[["slot", "p_dt", "p_sp", "p_ci", "family"]], on="slot", how="left")
t["Dn"] = t.D / np.sqrt(t.n)
pd.set_option("display.width", 250)
print(t.sort_values("bf_hier").head(20).round(3).to_string(index=False))
print("members with bf_hier < 5:", int((t.bf_hier < 5).sum()), "| D < 3:", int((t.D < 3).sum()), "| D missing:", int(t.D.isna().sum()))
t.to_parquet(f"{R3}/t36_member_table.parquet")
