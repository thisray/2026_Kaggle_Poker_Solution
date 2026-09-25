"""SP / DT listing-eligibility conditions (dev): partner's first preflop action after the first member, both see the flop,
members heads-up at some point, etc.  Recall on all evidence, pass rate on in-window non-evidence and on our wrong picks;
then E re-rank (hard filter) on the 20-candidate dev OOF pools."""
import numpy as np, pandas as pd
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
M = pd.read_parquet(f"{OUT}/t5_dev_seq.parquet")
W = pd.read_parquet(f"{OUT}/t4_wrong_vs_hit.parquet")[["slot", "h", "kind", "P_last_street_min", "R_hu_streets_sum", "R_opp_active_sum"]].rename(columns={"slot": "sl"})
M = M.merge(W, on=["sl", "h"], how="left")
conds = {
    "partner first action != fold": M.y2 != 0,
    "partner first action in (call,raise)": M.y2.isin([2, 3]),
    "first member action in (call,raise)": M.y1.isin([2, 3]),
    "both vpip (y1,y2 in call/raise/check)": M.y1.isin([1, 2, 3]) & M.y2.isin([1, 2, 3]),
    "both see flop": M.both_flop,
    "members heads-up >=1 street": M.R_hu_streets_sum.fillna(0) > 0,
}
for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]:
    F = M[(M.fam == fam) & (M.zone != "post")]
    print(f"== {fam}")
    for nm, c in conds.items():
        c = c.loc[F.index]
        print(f"   {nm:42s} recall(ev) {c[F.ev].mean():.3f} | pass(non) {c[~F.ev].mean():.3f} | pass(wrong_in) {c[F.kind == 'wrong_in'].mean():.3f} | pass(hit) {c[F.kind == 'hit'].mean():.3f}")
