"""R4-G3: listing rule of the two evidence sub-lists (DT). Type by hand pattern using omniscient equity at the sender's last street:
 A-pattern: S folds to R with S equity >= 0.6 (folds the better hand);  B-pattern: S equity <= 0.25, S put in >= 5bb, R wins, S did not fold before the flop.
Questions: (1) are all evidence hands ordered A-then-B by evidence_rank? (2) under 'A has priority, B fills to five', no A-pattern planted hand may be unlisted when n_A < 5:
compare the rate of strong A-pattern hands among NON-evidence hands of DT pairs with the same pattern rate in other families' pairs (control); same for B-pattern before / after the last listed B."""
import numpy as np, pandas as pd
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; pd.set_option("display.width", 250)
t5 = pd.read_parquet(f"{O}/t5_dev_seq.parquet").rename(columns={"sl": "slot"}); role = pd.read_parquet(f"{O}/r4/x2c_role_dev.parquet"); ker = pd.read_parquet(f"{O}/r4/x11_kernel_dev.parquet")
f = t5[["slot", "h", "fam", "ev", "ts"]].merge(role.drop(columns=["ts"]), on=["slot", "h"]).merge(ker, on=["slot", "h"]).sort_values(["slot", "ts", "h"]).reset_index(drop=True)
g1 = pd.read_parquet(f"{O}/r4/g1_evidence_rank.parquet")[["h", "pair_id", "evidence_rank", "chron"]]; f = f.merge(g1, on="h", how="left")
# orientation-free patterns (either member in the sender role)
for a, b, tag in (("s", "r", "SR"), ("r", "s", "RS")):
    eqa = f[f"k_p{a}_eq_last"]; eqb = f[f"k_p{b}_eq_last"]; fold_ab = f[f"x_{a}_fold_to_{b}"] if f"x_{a}_fold_to_{b}" in f else f["x_r_fold_to_s"]
    f["A_" + tag] = ((fold_ab == 1) & (eqa >= 0.6)).astype(int); f["As_" + tag] = ((fold_ab == 1) & (eqa >= 0.8) & (eqb <= 0.2)).astype(int)
    f["B_" + tag] = ((eqa <= 0.25) & (f[f"k_p{a}_contrib_bb"] >= 5) & (f[f"k_p{b}_won"] > 0) & (f[f"k_p{a}_last_street"] >= 1) & (f[f"k_p{b}_last_street"] >= 1)).astype(int)
f["A_any"] = f[["A_SR", "A_RS"]].max(axis=1); f["As_any"] = f[["As_SR", "As_RS"]].max(axis=1); f["B_any"] = f[["B_SR", "B_RS"]].max(axis=1)
dt = f[f.fam == "directed_transfer"].copy(); e = dt[dt.ev].copy(); e["type"] = np.where(e.A_SR == 1, "A", np.where(e.B_SR == 1, "B", "other"))
print("DT evidence by pattern type:", e.type.value_counts().to_dict())
viol = 0; pairs = 0; nA = []; 
for pid, g in e.sort_values("evidence_rank").groupby("pair_id"):
    t = g.type.values; pairs += 1; nA.append(int((t == "A").sum()))
    seenB = False
    for x in t:
        if x == "B": seenB = True
        if x == "A" and seenB: viol += 1; break
print(f"pairs {pairs}; pairs where an A-pattern hand is ranked after a B-pattern hand: {viol}; n_A distribution {pd.Series(nA).value_counts().sort_index().to_dict()}")
for typ in ("A", "B"):
    ch = e[e.type == typ].groupby("pair_id").apply(lambda g: bool((g.sort_values("evidence_rank").ts.diff().dropna() > 0).all()), include_groups=False); print(f"   within type {typ}: evidence_rank is chronological in {ch.mean():.3f} of pairs ({len(ch)})")
nA_map = e.groupby("slot").apply(lambda g: int((g.type == "A").sum()), include_groups=False); lastB = e[e.type == "B"].groupby("slot").ts.max(); lastA = e[e.type == "A"].groupby("slot").ts.max(); nev = e.groupby("slot").size()
dt["nA"] = dt.slot.map(nA_map).fillna(0); dt["nev"] = dt.slot.map(nev); dt["afterB"] = dt.ts > dt.slot.map(lastB).fillna(np.inf); dt["afterA"] = dt.ts > dt.slot.map(lastA).fillna(-np.inf)
ne = dt[~dt.ev]; ctrl = f[(f.fam != "directed_transfer") & (~f.ev)]
print(f"control (SP/CI pairs, non-evidence hands): A-pattern {ctrl.A_any.mean():.4f}  strong-A {ctrl.As_any.mean():.4f}  B-pattern {ctrl.B_any.mean():.4f}  (n={len(ctrl)})")
print(f"DT non-evidence, any direction:            A-pattern {ne.A_any.mean():.4f}  strong-A {ne.As_any.mean():.4f}  B-pattern {ne.B_any.mean():.4f}  (n={len(ne)})")
print(f"DT non-evidence, S->R direction only:      A-pattern {ne.A_SR.mean():.4f}  strong-A {ne.As_SR.mean():.4f}  B-pattern {ne.B_SR.mean():.4f}; evidence hands: A {e.A_SR.mean():.3f} strong-A {e.As_SR.mean():.3f} B {e.B_SR.mean():.3f}")
for lab, x in (("pairs with n_A<5 (list not filled by A)", ne[ne.nA < 5]), ("pairs with n_A==5", ne[ne.nA == 5])):
    print(f"   {lab}: A-pattern S->R rate {x.A_SR.mean():.4f}, strong-A {x.As_SR.mean():.4f} (n={len(x)}) | after the last listed A: A {x[x.afterA].A_SR.mean():.4f} strong-A {x[x.afterA].As_SR.mean():.4f} (n={int(x.afterA.sum())})")
full = ne[ne.nev == 5]; print(f"   B-pattern S->R among non-evidence of full lists: before last listed B {full[~full.afterB].B_SR.mean():.4f} (n={int((~full.afterB).sum())}) | after it {full[full.afterB].B_SR.mean():.4f} (n={int(full.afterB.sum())}) | control one-direction approx {ctrl.B_any.mean() / 2:.4f}")
f.to_parquet(f"{O}/r4/g3_patterns.parquet")
