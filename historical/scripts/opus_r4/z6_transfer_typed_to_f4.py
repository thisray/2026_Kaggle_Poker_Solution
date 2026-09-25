"""R4-Z6: transfer the DT two-type models to the fourth family. p_A / p_B are trained on dev DT (features without the t58 block: R18 gameplay + role + kernels, orientation by candidate flow),
then applied to every co-seated eval hand of the 87 'other' pairs in BOTH orientations (either member may be the giver); per hand p = 1 - (1 - p_orient1)(1 - p_orient2); A-priority decoder.
Outputs the top-5 per pair and the LB-consistency table (mean AP@5 of each historical submission's F4 evidence against this proxy, fitted K and SSE over the four paired LB deltas)."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, os, json
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18"); sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920")
import ci_censored_event as C
from g4_decode import decode, runs
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; C_ = f"{O}/r2_candidates"; NJ = 8; SRC = os.environ.get("SRC", "directed_transfer").split(",")
def assemble(frame, role, ker):
    ROLE = [c for c in role.columns if c.startswith("x_") and c not in ("x_k", "x_n", "x_flow_margin")]; KER = [c for c in ker.columns if c.startswith("k_")]
    d = frame.merge(role[["slot", "h"] + ROLE], on=["slot", "h"], validate="one_to_one").merge(ker[["slot", "h"] + KER], on=["slot", "h"], validate="one_to_one").sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True)
    return d, C.FEATURES + ROLE + KER
dev_all = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet")); dev_all = dev_all[dev_all.fam.isin(SRC)].reset_index(drop=True)
s, FS = assemble(dev_all, pd.read_parquet(f"{O}/r4/x2c_role_dev.parquet"), pd.read_parquet(f"{O}/r4/x11_kernel_dev.parquet"))
g1 = pd.read_parquet(f"{O}/r4/g1_evidence_rank.parquet")[["h", "pair_id", "evidence_rank", "chron"]].sort_values(["pair_id", "evidence_rank"]); g1["run"] = np.concatenate([runs(g.chron.values) for _, g in g1.groupby("pair_id", sort=False)]); g1["nruns"] = g1.groupby("pair_id").run.transform("max") + 1
s = s.merge(g1[["h", "run", "nruns"]], on="h", how="left").sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True); e = s[s.ev.astype(bool)]; two = e[e.nruns == 2]
clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=10, colsample_bytree=0.5, verbosity=-1, n_jobs=NJ).fit(two[FS].astype(float), (two.run == 1).astype(int))
typB = np.where(s.nruns.values == 2, s.run.values == 1, clf.predict_proba(s[FS].astype(float))[:, 1] > 0.5); s["isA"] = s.ev.astype(bool) & ~typB; s["isB"] = s.ev.astype(bool) & typB
nA = s.groupby("slot").isA.transform("sum"); nB = s.groupby("slot").isB.transform("sum"); nev = s.groupby("slot").ev.transform("sum"); lastA = s.slot.map(s[s.isA].groupby("slot").ts.max()); lastB = s.slot.map(s[s.isB].groupby("slot").ts.max())
incA = ((nA < 5) | (s.ts <= lastA)).values; incB = ((nev < 5) | ((nB > 0) & (s.ts <= lastB))).values
mA = [lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": NJ, "random_state": sd}).fit(s.loc[incA, FS].astype(float), s.loc[incA, "isA"]) for sd in (27, 28, 29)]; mB = [lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": NJ, "random_state": sd}).fit(s.loc[incB, FS].astype(float), s.loc[incB, "isB"]) for sd in (27, 28, 29)]
frame = C.prepare(pd.read_parquet(f"{O}/r4/y1_f4_eval_full.parquet")); P = {}
for tag in ("f4", "f4flip"):
    ev, FS2 = assemble(frame, pd.read_parquet(f"{O}/r4/x2_role_eval_{tag}.parquet"), pd.read_parquet(f"{O}/r4/x11_kernel_eval_{tag}.parquet")); assert FS == FS2
    P[tag] = (ev[["slot", "h"]].copy(), np.mean([m.predict_proba(ev[FS].astype(float))[:, 1] for m in mA], 0), np.mean([m.predict_proba(ev[FS].astype(float))[:, 1] for m in mB], 0))
assert (P["f4"][0].values == P["f4flip"][0].values).all(); ev = ev.reset_index(drop=True)
pA = 1 - (1 - P["f4"][1]) * (1 - P["f4flip"][1]); pB = 1 - (1 - P["f4"][2]) * (1 - P["f4flip"][2]); L, LA, LB = decode(ev, pA, pB)
LAM = float(os.environ.get("LAM", 0.0))
if LAM > 0:   # rule-based two-type decoder (orientation-free T3 patterns) blended in, as validated by the leave-one-family-out test (z8)
    sf = (ev.x_s_fold_to_r == 1) & (ev.k_pr_won > 0); rf = (ev.x_r_fold_to_s == 1) & (ev.k_ps_won > 0); A2 = ((sf & (ev.k_ps_eq_last >= 0.5)) | (rf & (ev.k_pr_eq_last >= 0.5))).values.astype(float)
    Bp = (((ev.k_ps_eq_last <= 0.3) & (ev.x_conS >= 5) & (ev.k_pr_won > 0) & (ev.x_s_fold_to_r == 0)) | ((ev.k_pr_eq_last <= 0.3) & (ev.x_conR >= 5) & (ev.k_ps_won > 0) & (ev.x_r_fold_to_s == 0))).values.astype(float)
    L = L + LAM * decode(ev, 0.6 * A2, 0.45 * Bp * (1 - A2))[0]
hid = pd.read_parquet(f"{O}/np/hand_index.parquet").set_index("hi").hand_id; ev["hand_id"] = ev.h.map(hid); ev["L"] = L; ev["pA"] = pA; ev["pB"] = pB; ev["LA"] = LA; ev["LB"] = LB
print(f"F4 pairs {ev.slot.nunique()}: sum pA per pair median {ev.groupby('slot').pA.sum().median():.2f}, sum pB {ev.groupby('slot').pB.sum().median():.2f} (dev DT positives: pA {s.groupby('slot').isA.sum().mean():.2f} listed A, {s.groupby('slot').isB.sum().mean():.2f} listed B)")
ev.to_parquet(f"{O}/r4/z6_f4_typed_transfer_{'_'.join(x[:2] for x in SRC)}.parquet")
top = ev.sort_values(["slot", "L", "ts"], ascending=[True, False, True]).groupby("pair_id").head(5).groupby("pair_id").hand_id.apply(list)
if os.environ.get("OUTPATCH"):
    import hashlib
    tp = ev.sort_values(["slot", "L", "ts", "h"], ascending=[True, False, True, True], kind="stable").groupby("pair_id", sort=False).head(5).copy(); tp["rank"] = tp.groupby("pair_id").cumcount() + 1
    pt = tp.pivot(index="pair_id", columns="rank", values="hand_id"); pt.columns = [f"evidence_hand_{i}" for i in pt.columns]; pt = pt.reset_index(); assert pt.notna().all().all(); pt.to_csv(os.environ["OUTPATCH"], index=False)
    json.dump(dict(source=SRC, pairs=len(pt), sha256=hashlib.sha256(open(os.environ["OUTPATCH"], "rb").read()).hexdigest()), open(os.environ["OUTPATCH"].replace(".csv", ".receipt.json"), "w"), indent=1); print("patch written", os.environ["OUTPATCH"], len(pt))
ndw = pd.read_csv(f"{C_}/r2n_NDw_on_r2j2m.csv", dtype=str); members = [p for p in ndw.pair_id[ndw.predicted_behavior == "other_coordination"] if p in top.index]
EVC = [f"evidence_hand_{i}" for i in range(1, 6)]
def ap5(pred, tru):
    hits = 0; sc = 0.0
    for i, p in enumerate(pred[:5]):
        if p in tru: hits += 1; sc += hits / (i + 1)
    return sc / min(5, len(tru))
subs = [("ND", "r2n_ND_on_r2j2m.csv", 0.91563), ("c-first", "r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv", 0.91838), ("NDdevpaw", "r2n_NDdevpaw_on_r2j2m.csv", 0.91974), ("NDw", "r2n_NDw_on_r2j2m.csv", 0.92303), ("subp", "r9_subp.csv", 0.91639), ("subh", "r9_subh.csv", 0.92115)]
ap = {nm: np.mean([ap5(list(pd.read_csv(f"{C_}/{f}", dtype=str, keep_default_na=False).set_index("pair_id").loc[p, EVC].values), set(top[p])) for p in members]) for nm, f, _ in subs}; lb = {nm: v for nm, _, v in subs}
dx = np.array([ap["c-first"] - ap["ND"], ap["NDdevpaw"] - ap["ND"], ap["NDw"] - ap["ND"], ap["subh"] - ap["subp"]]); dy = np.array([lb["c-first"] - lb["ND"], lb["NDdevpaw"] - lb["ND"], lb["NDw"] - lb["ND"], lb["subh"] - lb["subp"]])
K = float(dx @ dy / (dx @ dx)); sse = float(((dy - K * dx) ** 2).sum()); print("AP of historical submissions vs transfer-model top-5:", {k: round(v, 3) for k, v in ap.items()}); print(f"fitted K {K:.4f} (expected 0.026-0.033), SSE {sse:.2e}, residuals {np.round(dy - K * dx, 4).tolist()}")
