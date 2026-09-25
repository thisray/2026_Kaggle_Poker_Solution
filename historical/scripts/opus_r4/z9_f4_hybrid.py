"""R4-Z9: hybrid fourth-family slate. Type A ('a member folds the better hand to the partner') is the SAME event type in DT and SP, so its transfer to the fourth family is the reliable part;
the fourth family's type-B behaviour is unknown. Slate = confident type-A events first (L_A >= TAU, by L_A), then the current M3-ACT picks (r13) in their order, then remaining typed picks.
Reports LB-consistency (K, SSE over the four paired LB deltas) of each variant used as truth proxy, and writes patches."""
import numpy as np, pandas as pd, os, json, hashlib
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; C_ = f"{O}/r2_candidates"; EVC = [f"evidence_hand_{i}" for i in range(1, 6)]
ev = pd.read_parquet(f"{O}/r4/z6_f4_typed_transfer_di_so.parquet"); cur = pd.read_csv(f"{C_}/r13_ndwrank_cinew_f4.csv", dtype=str, keep_default_na=False).set_index("pair_id")
subs = [("ND", "r2n_ND_on_r2j2m.csv", 0.91563), ("c-first", "r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv", 0.91838), ("NDdevpaw", "r2n_NDdevpaw_on_r2j2m.csv", 0.91974), ("NDw", "r2n_NDw_on_r2j2m.csv", 0.92303), ("subp", "r9_subp.csv", 0.91639), ("subh", "r9_subh.csv", 0.92115)]
S = {nm: pd.read_csv(f"{C_}/{f}", dtype=str, keep_default_na=False).set_index("pair_id") for nm, f, _ in subs}; lb = {nm: v for nm, _, v in subs}; mem = [p for p in S["NDw"].index[S["NDw"].predicted_behavior == "other_coordination"] if p in set(ev.pair_id)]
dy = np.array([lb["c-first"] - lb["ND"], lb["NDdevpaw"] - lb["ND"], lb["NDw"] - lb["ND"], lb["subh"] - lb["subp"]])
def ap5(pred, tru):
    hits = 0; sc = 0.0
    for i, p in enumerate(pred[:5]):
        if p in tru: hits += 1; sc += hits / (i + 1)
    return sc / min(5, len(tru))
def slate(g, pid, tau):
    a = g[g.LA >= tau].sort_values("LA", ascending=False).hand_id.tolist()[:5]; rest = [h for h in cur.loc[pid, EVC].values if h not in a and h != "NO_EVIDENCE"]; fill = [h for h in g.sort_values("L", ascending=False).hand_id.tolist() if h not in a and h not in rest]
    return (a + rest + fill)[:5], len(a)
for tau in (0.3, 0.5, 0.7, 0.85, 2.0):
    SL = {}; na = []
    for pid, g in ev.groupby("pair_id"):
        SL[pid], k = slate(g, pid, tau); na.append(k)
    ap = {nm: np.mean([ap5(list(S[nm].loc[p, EVC].values), set(SL[p])) for p in mem]) for nm in S}; dx = np.array([ap["c-first"] - ap["ND"], ap["NDdevpaw"] - ap["ND"], ap["NDw"] - ap["ND"], ap["subh"] - ap["subp"]]); K = float(dx @ dy / (dx @ dx))
    ov = np.mean([len(set(SL[p]) & set(cur.loc[p, EVC].values)) for p in SL]); print(f"tau {tau}: confident A per pair {np.mean(na):.2f} | overlap with current {ov:.2f}/5 | K {K:.4f} SSE {((dy - K * dx) ** 2).sum():.2e} | AP(current r13 | this proxy) {np.mean([ap5(list(cur.loc[p, EVC].values), set(SL[p])) for p in mem]):.3f}")
    if tau in (0.5, 0.7):
        out = pd.DataFrame([[p] + SL[p] for p in SL], columns=["pair_id"] + EVC); path = f"{O}/r4/patch_r4_f4_hybrid_tau{str(tau).replace('.', '')}.csv"; out.to_csv(path, index=False)
        json.dump(dict(tau=tau, pairs=len(out), sha256=hashlib.sha256(open(path, "rb").read()).hexdigest()), open(path.replace(".csv", ".receipt.json"), "w"), indent=1)
