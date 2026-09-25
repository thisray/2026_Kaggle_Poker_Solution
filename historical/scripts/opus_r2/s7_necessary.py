"""Necessary-condition catalog: which pair-relation channels (R tensor) are ~always present in evidence hands, per family;
and does the r11 blend ever put structurally-impossible hands into its top 5?"""
import numpy as np, pandas as pd
A = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A}/opus_r1_20260917"; D_ = f"{OUT}/np"
RN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[0][2:].split(",")
R = np.load(f"{OUT}/R_v1.npy", mmap_mode="r"); sp = np.load(f"{D_}/s_player.npy")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
IND = ["facing", "fold_to", "call_to", "raise_over", "opp_active", "aggr_active", "passive_active", "squeeze", "iso_ofold", "hu_streets", "hu_noaggr", "check_hu", "bet_hu", "fold_hu_to"]
ci = [RN.index(c) for c in IND]
def rel(sl, h):
    plo = mem[sl // 900, (sl % 900) // 30]; phi = mem[sl // 900, sl % 30]
    sa = np.argmax(sp[h] == plo[:, None], axis=1); sb = np.argmax(sp[h] == phi[:, None], axis=1)
    ab = np.stack([np.asarray(R[h, sa, sb, c]) for c in ci], 1); ba = np.stack([np.asarray(R[h, sb, sa, c]) for c in ci], 1)
    return ab, ba
# ---- 1) catalog on all positive-pair dev hands
M = pd.read_parquet(f"{OUT}/m25e1_handfeat2_m19w10_oof.parquet").sort_values(["sl", "ts"]).reset_index(drop=True)
ab, ba = rel(M.sl.values, M.h.values)
last = M[M.ev].groupby("sl").ts.max(); M["win"] = M.ts <= M.sl.map(last)
anyd = (ab + ba) > 0; both = (ab > 0) & (ba > 0)
rows = []
for fm in ["directed_transfer", "soft_play", "coordinated_isolation"]:
    e = (M.fam == fm).values & M.ev.values; n = (M.fam == fm).values & ~M.ev.values & M.win.values
    for j, c in enumerate(IND):
        rows.append((fm[:2], c, anyd[e, j].mean(), anyd[n, j].mean(), both[e, j].mean(), both[n, j].mean()))
C = pd.DataFrame(rows, columns=["fam", "ch", "ev_any", "neg_any", "ev_both", "neg_both"]).round(3)
pd.set_option("display.width", 200); print(C.to_string(index=False))
# ---- 2) r11 top-5 violation of the best necessary condition
d = pd.read_parquet(f"{A}/round11_scoped/dev_oof_aligned.parquet")
n = pd.read_csv(f"{A}/round3_research_20260917/r6_narrow_candidates_v2.csv", usecols=["slot", "hand_id", "family", "hand_ts"])
d = d.merge(n, on=["slot", "hand_id"], how="left")
hidx = pd.read_parquet(f"{D_}/hand_index.parquet"); hmap = dict(zip(hidx.hand_id, hidx.hi)); d["h"] = d.hand_id.map(hmap).astype(np.int64)
ab2, ba2 = rel(d.slot.values, d.h.values); any2 = (ab2 + ba2) > 0
for j, c in enumerate(IND): d[f"a_{c}"] = any2[:, j]
d["brank"] = d.groupby("slot").rs_blend.rank(ascending=False, method="first")
print("\nfamily values:", d.family.value_counts().to_dict())
for c in ["facing", "opp_active", "aggr_active", "hu_streets"]:
    for fm, g in d.groupby("family"):
        top = g[g.brank <= 5]
        print(f"{fm[:2]} {c:12s} cand_rate {g[f'a_{c}'].mean():.3f}  ev_cov {g[g.ev == 1][f'a_{c}'].mean():.3f}  top5_violation {1 - top[f'a_{c}'].mean():.3f}  top5_viol_nonev {((~top[f'a_{c}']) & (top.ev == 0)).mean():.3f}")
def ap5(g, col):
    top = g.sort_values(col, ascending=False, kind="mergesort").ev.values[:5]; hits = 0; s = 0.0
    for i, e in enumerate(top):
        if e: hits += 1; s += hits / (i + 1)
    return s / min(5, int(g.m_p.iloc[0]))
def E(df, col): return float(np.mean([ap5(g, col) for _, g in df.groupby("slot")]))
base = E(d, "rs_blend"); print("\nbase E", round(base, 6))
for c in ["facing", "opp_active", "aggr_active"]:
    for fams in [("directed_transfer", "soft_play"), ("directed_transfer", "soft_play", "coordinated_isolation")]:
        dem = d.family.isin(fams) & ~d[f"a_{c}"]
        d["tmp"] = d.rs_blend - 10 * dem
        print(f"demote non-{c} for {[f[:2] for f in fams]}: E {E(d, 'tmp'):.6f} ({E(d, 'tmp') - base:+.6f})")
