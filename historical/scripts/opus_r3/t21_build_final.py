"""Assemble an R3 candidate from the r2j2m base (LB-verified P/B of NDw 0.92303):
  PROMOTE  : pair_ids moved right after the member block (in the given order) and relabelled other_coordination (B6-style)
  DEMOTE   : member pair_ids relabelled to the m7 family (rank unchanged), evidence = r15 top-5 (CI filter if CI)
  F4 EV    : NDw_sub (substitution-mechanism posteriors, t17 rows) for every other_coordination pair, or keep tilt NDw
  CI patch : r3/t20_ci_patch.parquet for CI-predicted pairs (optional)
Env: OUTNAME, PROMOTE (comma list or file), DEMOTE, F4EV in {sub, tilt}, CIPATCH in {0,1}."""
import numpy as np, pandas as pd, hashlib, json, os
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"; R3 = f"{OUT}/r3"
def lst(v):
    if not v: return []
    if os.path.exists(v): return [x.strip() for x in open(v).read().split() if x.strip()]
    return [x for x in v.split(",") if x]
OUTNAME = os.environ["OUTNAME"]; PROMOTE = lst(os.environ.get("PROMOTE", "")); DEMOTE = lst(os.environ.get("DEMOTE", "")); RELABEL = lst(os.environ.get("RELABEL", "")); KEEP = set(lst(os.environ.get("PROMOTE_KEEP", "")))   # promoted in rank but keep family label and evidence
F4EV = os.environ.get("F4EV", "sub"); CIPATCH = os.environ.get("CIPATCH", "1") == "1"
base = pd.read_csv(f"{C}/r2n_NDw_on_r2j2m.csv", dtype=str)                       # risk/behaviour of r2j2m, tilt-NDw F4 evidence
EVC = [f"evidence_hand_{i}" for i in range(1, 6)]
r = base.risk_score.astype(float).values; order0 = base.pair_id.values[np.argsort(-r, kind="stable")]; pos0 = pd.Series(np.arange(len(order0)), index=order0)
mem = set(base[base.predicted_behavior == "other_coordination"].pair_id)
assert not (set(PROMOTE) & mem) and set(DEMOTE) <= mem and not (set(RELABEL) & mem) and not (set(RELABEL) & set(PROMOTE))
last_member_pos = int(pos0.loc[list(mem)].max())
prom = set(PROMOTE); rest = [p for p in order0 if p not in prom]
ins = rest.index(order0[last_member_pos]) + 1
new_order = rest[:ins] + PROMOTE + rest[ins:]; N = len(new_order); risk = pd.Series((N - np.arange(N)) / N, index=new_order)
out = base.set_index("pair_id").copy(); out["risk_score"] = out.index.map(risk).map(lambda v: repr(float(v)))
out.loc[[p for p in PROMOTE if p not in KEEP] + RELABEL, "predicted_behavior"] = "other_coordination"
m7 = pd.read_parquet(f"{OUT}/m7_family_eval.parquet"); e85 = pd.read_parquet(f"{OUT}/s85_eval_bf.parquet")[["slot", "pair_id"]]
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
lo = m7.key // 12000; hi = m7.key % 12000; m7["slot"] = loc.pool.loc[lo].values * 900 + loc.local.loc[lo].values * 30 + loc.local.loc[hi].values
fam_of = m7.merge(e85, on="slot").set_index("pair_id").family
for pid in DEMOTE: out.loc[pid, "predicted_behavior"] = fam_of[pid]
# evidence
sc = pd.read_csv(f"{A_}/round15_campaign/scored_tabicl_rank_blend.csv")
patch = pd.read_parquet(f"{R3}/t20_ci_patch.parquet").set_index("pair_id") if CIPATCH else None
for pid in DEMOTE:
    g = sc[sc.pair_id == pid].sort_values("score", ascending=False, kind="mergesort")
    assert len(g) >= 5, pid
    out.loc[pid, EVC] = g.hand_id.values[:5]
if CIPATCH:
    ci = out[out.predicted_behavior == "coordinated_isolation"].index
    idx = [p for p in patch.index if p in set(ci)]; out.loc[idx, EVC] = patch.loc[idx, EVC].values
    # demoted CI pairs: recompute with the filter if present in patch
oth = out[out.predicted_behavior == "other_coordination"].index.tolist()
if F4EV in ("sub", "subh", "subp", "mix", "witness"):
    pid2sl = e85.set_index("pair_id").slot; slots = pid2sl.loc[oth].values
    if F4EV == "sub":
        rows = pd.read_parquet(f"{R3}/t17_rows_eval.parquet"); rows = rows[rows.slot.isin(slots)]
        hq = rows.groupby(["slot", "h"]).post.apply(lambda p: 1 - np.prod(1 - p.values)).rename("q").reset_index()
    else:
        hq = pd.read_parquet(f"{R3}/t17_hands_eval.parquet")[["slot", "h", "q_act", "q_plant"]]; hq = hq[hq.slot.isin(slots)]
        hq["q"] = hq.q_act if F4EV == "subh" else hq.q_plant
        if F4EV == "witness":
            wq = pd.read_parquet(f"{OUT}/r18/f4_witness.parquet")[["slot", "h", "q_witness"]]; hq = hq.merge(wq, on=["slot", "h"], how="left"); hq["q"] = hq.q_witness.fillna(0.0)
    H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, slots); H, S, T, SL = H[m], S[m], T[m], SL[m]
    won = np.load(f"{D}/s_won.npy", mmap_mode="r"); ts = np.load(f"{D}/h_ts.npy"); W = np.asarray(won[H]); ix = np.arange(len(H))
    G = pd.DataFrame({"slot": SL, "h": H, "ts": ts[H], "pw": (W[ix, S] > 0) | (W[ix, T] > 0)}).merge(hq, on=["slot", "h"], how="left").fillna({"q": 0.709 if F4EV == "subp" else 0.0, "q_act": 0.0, "q_plant": 0.709})   # planted prior rho for hands without member decisions
    G = G.sort_values(["slot", "ts", "h"]); groups = {s: g for s, g in G.groupby("slot")}
    def first_k_prob(q, K=5):
        o = np.zeros(len(q)); dist = np.zeros(K + 1); dist[0] = 1.0
        for i, p in enumerate(q):
            o[i] = p * dist[:K].sum(); nd = dist * (1 - p); nd[1:] += dist[:-1] * p; nd[K] += dist[K] * p; dist = nd
        return o
    hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hi2id = dict(zip(hidx.hi, hidx.hand_id))
    for pid in oth:
        g = groups[pid2sl[pid]]
        if F4EV == "mix": s_ = first_k_prob(g.q_act.values * g.pw.values) + first_k_prob(g.q_plant.values * g.pw.values) + 1e-6 * first_k_prob(g.q_act.values)
        else: s_ = first_k_prob(g.q.values * g.pw.values) + 1e-6 * first_k_prob(g.q.values)
        p = g.h.values[np.argsort(-s_, kind="stable")][:5]; assert len(set(p)) == 5
        out.loc[pid, EVC] = [hi2id[x] for x in p]
else:
    tilt = pd.read_csv(f"{C}/r2f_NDw_all_on_r2j2mB6.csv", dtype=str).set_index("pair_id")      # tilt NDw for all 85 other pairs
    miss = [p for p in oth if p not in tilt.index or tilt.loc[p, "predicted_behavior"] != "other_coordination"]
    assert not miss, f"tilt NDw not available for {len(miss)} pairs"
    out.loc[oth, EVC] = tilt.loc[oth, EVC].values
o = out.reset_index()[base.columns]; path = f"{C}/{OUTNAME}.csv"; o.to_csv(path, index=False)
rec = dict(file=OUTNAME + ".csv", sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(), promote=PROMOTE, promote_keep=sorted(KEEP), relabel=RELABEL, demote=DEMOTE, f4ev=F4EV, cipatch=CIPATCH,
           n_other=len(oth), risk_changed=int((o.risk_score.values != base.risk_score.values).sum()), behavior_changed=int((o.predicted_behavior.values != base.predicted_behavior.values).sum()),
           evidence_rows_changed=int((o[EVC].values != base[EVC].values).any(1).sum()))
json.dump(rec, open(path.replace(".csv", ".receipt.json"), "w"), indent=1); print(json.dumps(rec))
