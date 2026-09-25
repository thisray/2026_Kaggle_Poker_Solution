"""Meta-validation of 'earliest clear deviation' evidence rules on KNOWN families (true evidence available):
DT clear deviation = a member folds to the partner's bet/raise with heads-up omniscient equity >= 0.7 and value flows to the
partner; SP clear deviation = heads-up with the partner, holding >= 0.8 equity, checks (no bet) ; CI clear deviation = both
members aggressive and an outsider folds (iso_ofold).  Rule: earliest 5 such hands among ALL pair hands (dev phase), filled
with the next earliest weaker events.  Compared with the learned r15 stack on the same pairs."""
import numpy as np, pandas as pd
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D_ = f"{OUT}/np"
L = open(f"{OUT}/feature_names_v1.txt").read().split("\n"); RN = L[0][2:].split(","); PN = L[1][2:].split(",")
R = np.load(f"{OUT}/R_v1.npy", mmap_mode="r"); sp = np.load(f"{D_}/s_player.npy")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
M = pd.read_parquet(f"{OUT}/m25e1_handfeat2_m19w10_oof.parquet")[["sl", "h", "ev", "fam", "ts"]].sort_values(["sl", "ts"]).reset_index(drop=True)
plo = mem[M.sl.values // 900, (M.sl.values % 900) // 30]; phi = mem[M.sl.values // 900, M.sl.values % 30]
sa = np.argmax(sp[M.h.values] == plo[:, None], axis=1); sb = np.argmax(sp[M.h.values] == phi[:, None], axis=1)
g = lambda i, j, c: np.asarray(R[M.h.values, i, j, RN.index(c)])
fab, fba = g(sa, sb, "fold_to"), g(sb, sa, "fold_to"); eab, eba = g(sa, sb, "eq_fold_to"), g(sb, sa, "eq_fold_to")
flab, flba = g(sa, sb, "flow"), g(sb, sa, "flow")
M["dt_clear"] = (((fab > 0) & (eab / np.maximum(fab, 1) >= 0.7) & (flab > 0)) | ((fba > 0) & (eba / np.maximum(fba, 1) >= 0.7) & (flba > 0))).astype(int)
M["dt_weak"] = (((flab > 0) | (flba > 0)) & ((g(sa, sb, "facing") + g(sb, sa, "facing")) > 0)).astype(int)
chk_ab, chk_ba = g(sa, sb, "check_hu"), g(sb, sa, "check_hu"); echk_ab, echk_ba = g(sa, sb, "eq_check_hu"), g(sb, sa, "eq_check_hu")
M["sp_clear"] = (((chk_ab > 0) & (echk_ab / np.maximum(chk_ab, 1) >= 0.8)) | ((chk_ba > 0) & (echk_ba / np.maximum(chk_ba, 1) >= 0.8))).astype(int)
M["sp_weak"] = (((g(sa, sb, "facing") + g(sb, sa, "facing")) > 0) & ((g(sa, sb, "passive_active") > 0) & (g(sb, sa, "passive_active") > 0))).astype(int)
M["ci_clear"] = ((g(sa, sb, "iso_ofold") + g(sb, sa, "iso_ofold")) > 0).astype(int)
M["ci_weak"] = ((g(sa, sb, "aggr_active") > 0) & (g(sb, sa, "aggr_active") > 0)).astype(int)
def ap5(picks, evset, m):
    hits = 0; s = 0.0
    for i, h in enumerate(picks[:5]):
        if h in evset: hits += 1; s += hits / (i + 1)
    return s / min(5, m)
res = {}
for fam, pre in [("directed_transfer", "dt"), ("soft_play", "sp"), ("coordinated_isolation", "ci")]:
    F = M[M.fam == fam]; aps = []; cov = []
    for sl, gg in F.groupby("sl"):
        evs = set(gg[gg.ev].h); m = len(evs)
        p = gg[gg[f"{pre}_clear"] == 1].h.tolist()
        p += [x for x in gg[gg[f"{pre}_weak"] == 1].h.tolist() if x not in p]
        aps.append(ap5(p, evs, m)); cov.append(gg[gg.ev][f"{pre}_clear"].mean())
    res[fam] = (np.mean(aps), np.mean(cov))
    print(f"{fam[:2]}: 'earliest clear deviation' rule E = {np.mean(aps):.4f}; share of evidence that is a clear deviation = {np.mean(cov):.3f}")
print("reference learned r15 E (dev): DT 0.715, SP 0.720, CI 0.654")
