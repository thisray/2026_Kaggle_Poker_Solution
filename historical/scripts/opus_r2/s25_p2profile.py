"""Behavioural profile of eval 'pattern-2' pairs vs known families (dev positives, eval family-predicted pairs) using per-hand R/P channels."""
import numpy as np, pandas as pd
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
L = open(f"{OUT}/feature_names_v1.txt").read().split("\n"); RN = L[0][2:].split(","); PN = L[1][2:].split(",")
R = np.load(f"{OUT}/R_v1.npy", mmap_mode="r"); Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r")
e = pd.read_parquet(f"{OUT}/s23_infoshare_eval.parquet").merge(pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv"), on="slot")
sub = pd.read_csv(f"{A_}/round11_scoped/r11_scoped.csv", usecols=["pair_id", "predicted_behavior"]); e = e.merge(sub, on="pair_id")
e["rk"] = e.risk_score.rank(ascending=False); e["p2"] = np.minimum(e.za0 - e.zf0, e.za1 - e.zf1); e["dt"] = np.maximum(e.zf0 - e.za0, e.zf1 - e.za1)
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); lp = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
dv["slot"] = PI.pair_slot(dv.pool.values, lp.local.loc[dv.p_lo].values, lp.local.loc[dv.p_hi].values)
rng = np.random.default_rng(0)
groups = {
    "E p2>4 top450": (1, e[(e.rk <= 450) & (e.p2 > 4)].slot.values),
    "E p2>4 rk>450": (1, e[(e.rk > 450) & (e.p2 > 4)].slot.values),
    "E CI-pred top450 p2<1": (1, e[(e.rk <= 450) & (e.p2 < 1) & (e.predicted_behavior == "coordinated_isolation")].slot.values),
    "E DT-pred top450": (1, e[(e.rk <= 450) & (e.predicted_behavior == "directed_transfer")].slot.values),
    "E SP-pred top450": (1, e[(e.rk <= 450) & (e.predicted_behavior == "soft_play")].slot.values),
    "E random rk>5000": (1, rng.choice(e[e.rk > 5000].slot.values, 1500, replace=False)),
    "D CI pos": (0, dv[(dv.label == 1) & (dv.fam == "coordinated_isolation")].slot.values),
    "D DT pos": (0, dv[(dv.label == 1) & (dv.fam == "directed_transfer")].slot.values),
    "D SP pos": (0, dv[(dv.label == 1) & (dv.fam == "soft_play")].slot.values),
}
cache = {ph: PI.all_pair_hands(ph) for ph in [0, 1]}
ri = {c: RN.index(c) for c in ["facing", "fold_to", "call_to", "raise_over", "squeeze", "iso_ofold", "hu_streets", "flow", "aggr_active", "bet_hu", "check_hu"]}
pi = {c: PN.index(c) for c in ["vpip", "pfr", "sd", "won", "net_bb", "contrib_bb"]}
rows = []
for nm, (ph, slots) in groups.items():
    H, S, T, SL = cache[ph]; m = np.isin(SL, slots); h, a, b = H[m], S[m], T[m]
    out = {"group": nm, "pairs": len(slots), "hands/pair": round(m.sum() / max(len(slots), 1), 1)}
    va = np.asarray(Pt[h, a, pi["vpip"]]); vb = np.asarray(Pt[h, b, pi["vpip"]]); both = (va > 0) & (vb > 0); out["both_vpip"] = both.mean()
    pa = np.asarray(Pt[h, a, pi["pfr"]]); pb = np.asarray(Pt[h, b, pi["pfr"]]); out["both_pfr|both"] = ((pa > 0) & (pb > 0))[both].mean()
    for c in ["raise_over", "squeeze", "iso_ofold", "fold_to", "call_to", "hu_streets", "flow"]:
        x = (np.asarray(R[h, a, b, ri[c]]) + np.asarray(R[h, b, a, ri[c]])) > 0; out[c + "|both"] = x[both].mean()
    sda = np.asarray(Pt[h, a, pi["sd"]]); sdb = np.asarray(Pt[h, b, pi["sd"]]); wa = np.asarray(Pt[h, a, pi["won"]]); wb = np.asarray(Pt[h, b, pi["won"]])
    na = np.asarray(Pt[h, a, pi["net_bb"]]); nb = np.asarray(Pt[h, b, pi["net_bb"]])
    out["one_sd_other_folded|both"] = (((sda > 0) & (sdb == 0)) | ((sdb > 0) & (sda == 0)))[both].mean()
    out["pair_net_bb|both"] = (na + nb)[both].mean(); out["pair_net_bb_all"] = (na + nb).mean()
    out["big_win(net>20bb)|both"] = (np.maximum(na, nb) > 20)[both].mean()
    rows.append(out)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)
print(pd.DataFrame(rows).round(3).to_string(index=False))
