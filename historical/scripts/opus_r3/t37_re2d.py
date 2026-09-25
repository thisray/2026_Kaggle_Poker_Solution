"""R3-F15: which fourth-family rate is pair-specific - the hand planting rate rho, the decision usage rate s, or both?

Model (per pair p): Z_h ~ Bern(rho_p); given Z_h = 1 each member decision uses the partner's cards with
s_d = sigmoid(logit(s_street) + sigma * u_p), u_p ~ N(0, 1); rho_p ~ Beta(mu * kappa, (1 - mu) * kappa).
  M0 fixed rho, fixed s | M1 random rho (R19) | M2 random s | M3 random rho and s.
Pool-grouped 5-fold held-out log-likelihood (4 seeds, as R19), then first-5 decoders that integrate the pair posterior
after the DP, and a self-consistent simulation (truth from the M3 / M1 posterior) scoring fixed / M1 / M3 pickers
under the ACT and PLANT labeller rules. Writes E-only candidates for the M3 decoders."""
import os, sys, json, hashlib
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp, roots_jacobi
from numpy.polynomial.hermite_e import hermegauss
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pairindex as PI
from f4_random_effects import first_k_at_nodes

A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"; R3 = f"{OUT}/r3"
BASE = os.environ.get("BASE", "r5_subh.csv"); PREFIX = os.environ.get("PREFIX", "r6_re2"); NS = int(os.environ.get("NS", "300"))
NR, NU = int(os.environ.get("NR", "32")), int(os.environ.get("NU", "12"))
WIT = os.environ.get("WIT") == "1"; OUTS = [x.split(":") for x in os.environ.get("OUTS", "m3_act:subh,m3_plant:subp,m3_mix:mix").split(",")]
RHO0, SPRE, SPOST = 0.70919, 0.50769, 0.42580
EVC = [f"evidence_hand_{i}" for i in range(1, 6)]
base = pd.read_csv(f"{C}/{BASE}", dtype=str); bidx = base.set_index("pair_id")
oth = bidx.index[bidx.predicted_behavior == "other_coordination"].tolist()
e85 = pd.read_parquet(f"{OUT}/s85_eval_bf.parquet")[["slot", "pair_id"]].set_index("pair_id").slot
slots = e85.loc[oth].values; sl2pid = dict(zip(slots, oth))
H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, slots); H, S, T, SL = H[m], S[m], T[m], SL[m]
won = np.load(f"{D}/s_won.npy", mmap_mode="r"); ts = np.load(f"{D}/h_ts.npy"); W = np.asarray(won[H]); ix = np.arange(len(H))
G = pd.DataFrame({"slot": SL, "h": H, "ts": ts[H], "pw": ((W[ix, S] > 0) | (W[ix, T] > 0)).astype(float)}).sort_values(["slot", "ts", "h"]).reset_index(drop=True)
G["hid"] = np.arange(len(G)); G["pool"] = G.slot // 900
rows = pd.read_parquet(f"{R3}/t17_rows_eval.parquet", columns=["k", "h", "st", "slot", "r"]); rows = rows[rows.slot.isin(slots)]
OOT = os.environ.get("OOT") == "1"
if OOT:   # policy_v2 memorised its 6M training rows (t39: member log q0 gap +0.43): those decisions carry no usable ratio -> r := 1
    NDEC = len(np.load(f"{OUT}/dec_Y.npy", mmap_mode="r")); intrain = np.zeros(NDEC, bool)
    intrain[np.random.RandomState(1).choice(NDEC, 6_000_000, replace=False)] = True
    if os.environ.get("RV1") == "1":   # t40: policy_v1 ratio where v1 did not train on the decision; r := 1 only if both models trained on it
        v1 = pd.read_parquet(f"{R3}/t40_v1_ratio.parquet")[["k", "slot", "r_v1", "in_v1"]]; rows = rows.merge(v1, on=["k", "slot"], how="left")
        assert rows.r_v1.notna().all(), "t40 must cover every member row"
        use1 = intrain[rows.k.values] & ~rows.in_v1.values.astype(bool); both = intrain[rows.k.values] & rows.in_v1.values.astype(bool)
        rows.loc[use1, "r"] = rows.r_v1[use1]; rows.loc[both, "r"] = 1.0; rows = rows.drop(columns=["r_v1", "in_v1"])
        print(f"OOT+RV1: {int(use1.sum())} rows use policy_v1 ratio, {int(both.sum())} of {len(rows)} uninformative")
    else:
        rows.loc[intrain[rows.k.values], "r"] = 1.0; print(f"OOT: {int(intrain[rows.k.values].sum())} of {len(rows)} member rows set uninformative")
rows = rows.merge(G[["slot", "h", "hid"]], on=["slot", "h"], how="inner").sort_values("hid").reset_index(drop=True)
hq = pd.read_parquet(f"{R3}/t17_hands_eval.parquet")[["slot", "h", "q_act", "q_plant"]]
G = G.merge(hq, on=["slot", "h"], how="left").fillna({"q_act": 0.0, "q_plant": 0.709}).sort_values("hid").reset_index(drop=True)
print(f"{len(oth)} pairs, {len(G)} hands, {len(rows)} member decisions")
r_all = rows.r.values.clip(1e-12, None); pre_all = (rows.st.values == 0); hid_all = rows.hid.values

def nodes(model, th):
    lsp, lspo, lmu, lka, lsg = th
    if model in ("M1", "M3"):
        mu, ka = expit(lmu), np.exp(lka); x, w = roots_jacobi(NR, (1 - mu) * ka - 1, mu * ka - 1); rho = (x + 1) / 2; lwr = np.log(w) - logsumexp(np.log(w))
    else:
        rho = np.array([expit(lmu)]); lwr = np.zeros(1)
    if model in ("M2", "M3"):
        x, w = hermegauss(NU); u = np.exp(lsg) * x; lwu = np.log(w) - logsumexp(np.log(w))
    else:
        u = np.zeros(1); lwu = np.zeros(1)
    return rho.clip(1e-9, 1 - 1e-9), lwr, u, lwu

def hand_terms(th, u, sel_rows):
    """lf, l0 per hand (only hands with decisions) for each u node: arrays (n_hands_sel, NU)."""
    lsp, lspo = th[0], th[1]; r = r_all[sel_rows]; pre = pre_all[sel_rows]; hid = hid_all[sel_rows]
    s = expit(np.where(pre, lsp, lspo)[:, None] + u[None, :])
    t1 = np.log1p(s * (r[:, None] - 1)); t0 = np.log1p(-s)
    starts = np.r_[0, np.flatnonzero(np.diff(hid)) + 1]
    if WIT:   # witness: decision active AND action changed (maximal coupling): u_d = s max(r-1, 0) / (1 - s + s r)
        tw = np.log1p(-(s * np.maximum(r[:, None] - 1, 0) / (1 - s + s * r[:, None])).clip(0, 1 - 1e-12))
        return np.add.reduceat(t1, starts, axis=0), np.add.reduceat(t0, starts, axis=0), hid[starts], np.add.reduceat(tw, starts, axis=0)
    return np.add.reduceat(t1, starts, axis=0), np.add.reduceat(t0, starts, axis=0), hid[starts]

def pair_ll(model, th, sel_rows):
    rho, lwr, u, lwu = nodes(model, th)
    lf, _, hids = hand_terms(th, u, sel_rows)[:3]
    loc = np.logaddexp(np.log1p(-rho)[None, None, :], np.log(rho)[None, None, :] + lf[:, :, None])   # hands x U x R
    slot_of = G.slot.values[hids]; starts = np.r_[0, np.flatnonzero(np.diff(slot_of)) + 1]
    lp = np.add.reduceat(loc, starts, axis=0)                                                          # pairs x U x R
    return logsumexp(lp + lwu[None, :, None] + lwr[None, None, :], axis=(1, 2)), slot_of[starts], (rho, lwr, u, lwu)

TH0 = np.array([logit(SPRE), logit(SPOST), logit(RHO0), np.log(2.5), np.log(0.5)])
FITLOG = []
FREE = {"M0": [0, 1, 2], "M1": [0, 1, 2, 3], "M2": [0, 1, 2, 4], "M3": [0, 1, 2, 3, 4]}
def fit(model, sel_rows):
    fr = FREE[model]
    def f(x):
        th = TH0.copy(); th[fr] = x; return -pair_ll(model, th, sel_rows)[0].sum()
    r = minimize(f, TH0[fr], method="L-BFGS-B", bounds=[((-4, 4) if i < 3 else ((-2, 6) if i == 3 else (-4, 2))) for i in fr],
                 options={"maxiter": 200})
    assert np.isfinite(r.fun) and np.isfinite(r.x).all(), (model, r.message)
    FITLOG.append(dict(model=model, n_rows=int(np.sum(sel_rows)), success=bool(r.success), message=str(r.message), nit=int(r.nit), ll=float(-r.fun)))
    if not r.success: print(f"WARNING fit {model}: {r.message}", flush=True)
    th = TH0.copy(); th[fr] = r.x; return th, -r.fun

rows_pool = G.pool.values[hid_all]
full = {}
for mdl in ("M0", "M1", "M2", "M3"):
    th, ll = fit(mdl, np.ones(len(rows), bool)); full[mdl] = th
    print(f"full {mdl}: LL {ll:.3f} s_pre {expit(th[0]):.4f} s_post {expit(th[1]):.4f} mu {expit(th[2]):.4f} kappa {np.exp(th[3]):.3f} sigma_u {np.exp(th[4]):.3f}", flush=True)
pools = np.unique(G.pool.values); cv = []
for seed in (() if os.environ.get("SKIPCV") == "1" else (260919, 11, 29, 47)):
    rs = np.random.default_rng(seed); perm = rs.permutation(pools); fold_of = {p: i % 5 for i, p in enumerate(perm)}; fr_ = np.array([fold_of[p] for p in rows_pool])
    tot = {m_: 0.0 for m_ in full}
    for f in range(5):
        tr, te = fr_ != f, fr_ == f
        for mdl in full:
            th, _ = fit(mdl, tr); tot[mdl] += pair_ll(mdl, th, te)[0].sum()
    cv.append(dict(seed=seed, **tot)); print("cv", seed, {k: round(v, 3) for k, v in tot.items()}, "| M1-M0", round(tot["M1"] - tot["M0"], 2), "M2-M0", round(tot["M2"] - tot["M0"], 2), "M3-M1", round(tot["M3"] - tot["M1"], 2), flush=True)

def first_k_prob(q, K=5):
    o = np.zeros(len(q)); dist = np.zeros(K + 1); dist[0] = 1.0
    for i, p in enumerate(q):
        o[i] = p * dist[:K].sum(); nd = dist * (1 - p); nd[1:] += dist[:-1] * p; nd[K] += dist[K] * p; dist = nd
    return o
def top5(score, hs):
    p = hs[np.argsort(-score, kind="stable")][:5]; assert len(set(p)) == 5; return p

# per-pair posterior over (u, rho) nodes and per-hand lf/l0 at every u node, for M1 and M3
post = {}
for mdl in ("M1", "M3"):
    th = full[mdl]; lpp, sl_order, (rho, lwr, u, lwu) = pair_ll(mdl, th, np.ones(len(rows), bool))
    ht_ = hand_terms(th, u, np.ones(len(rows), bool)); lf, l0, hids = ht_[:3]
    LF = np.zeros((len(G), len(u))); L0 = np.zeros((len(G), len(u))); LW = np.zeros((len(G), len(u))); LF[hids] = lf; L0[hids] = l0
    if WIT: LW[hids] = ht_[3]
    loc = np.logaddexp(np.log1p(-rho)[None, None, :], np.log(rho)[None, None, :] + LF[:, :, None])
    post[mdl] = dict(th=th, rho=rho, lwr=lwr, u=u, lwu=lwu, LF=LF, L0=L0, LW=LW, loc=loc)
    if mdl == "M3":   # membership diagnostic: pair marginal log-likelihood ratio (substitution model vs normal play)
        pl = pd.DataFrame({"slot": sl_order, "llr_m3": lpp}); pl["pair_id"] = pl.slot.map(sl2pid)
        pl.to_parquet(f"{R3}/t37_pairllr_{PREFIX}.parquet"); print("lowest M3 pair LLR:", pl.nsmallest(8, "llr_m3")[["pair_id", "llr_m3"]].round(2).values.tolist(), flush=True)
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hi2id = dict(zip(hidx.hi, hidx.hand_id))
names = ("fix_act", "fix_plant", "m1_act", "m1_plant", "m3_act", "m3_plant", "m3_mix") + (("m3_wit",) if WIT else ()); picks = {k: {} for k in names}
RULES = ("ACT", "PLANT") + (("WITNESS",) if WIT else ())
sim = {(tr_, rule, k): [] for tr_ in ("M3", "M1") for rule in RULES for k in names}; rng = np.random.default_rng(20260920); info = []
for sl, g in G.groupby("slot", sort=False):
    pid = sl2pid[sl]; hs = g.h.values; hh = g.hid.values; pw = g.pw.values
    picks["fix_act"][pid] = top5(first_k_prob(g.q_act.values * pw) + 1e-6 * first_k_prob(g.q_act.values), hs)
    picks["fix_plant"][pid] = top5(first_k_prob(g.q_plant.values * pw) + 1e-6 * first_k_prob(g.q_plant.values), hs)
    cache = {}
    for mdl, tag in (("M1", "m1"), ("M3", "m3")):
        P_ = post[mdl]; rho, u = P_["rho"], P_["u"]
        lw = P_["loc"][hh].sum(0) + P_["lwu"][:, None] + P_["lwr"][None, :]; w = np.exp(lw - lw.max()); w /= w.sum()   # U x R
        LF, L0 = P_["LF"][hh], P_["L0"][hh]                                                                       # hands x U
        qp = expit(logit(rho)[None, None, :] + LF[:, :, None]); qa = qp * (-np.expm1(np.minimum(L0 - LF, 0)))[:, :, None]
        qp2, qa2, w2 = qp.reshape(len(hs), -1), qa.reshape(len(hs), -1), w.reshape(-1)
        keep = w2 > 1e-10; qp2, qa2, w2 = qp2[:, keep], qa2[:, keep], w2[keep] / w2[keep].sum()
        sa = first_k_at_nodes(qa2 * pw[:, None]) @ w2 + 1e-6 * (first_k_at_nodes(qa2) @ w2)
        sp_ = first_k_at_nodes(qp2 * pw[:, None]) @ w2 + 1e-6 * (first_k_at_nodes(qp2) @ w2)
        picks[f"{tag}_act"][pid] = top5(sa, hs); picks[f"{tag}_plant"][pid] = top5(sp_, hs)
        if tag == "m3": picks["m3_mix"][pid] = top5(sa + sp_, hs)
        if tag == "m3" and WIT:
            qw2 = (qp * (-np.expm1(P_["LW"][hh]))[:, :, None]).reshape(len(hs), -1)[:, keep]
            picks["m3_wit"][pid] = top5(first_k_at_nodes(qw2 * pw[:, None]) @ w2 + 1e-6 * (first_k_at_nodes(qw2) @ w2), hs)
        cache[mdl] = (qp2, w2, keep, u, rho)
        if mdl == "M3":
            uu = np.repeat(u, len(rho))[keep]; rr_ = np.tile(rho, len(u))[keep]
            info.append(dict(pair_id=pid, rho_mean=float(rr_ @ w2), u_mean=float(uu @ w2), hands=len(hs)))
    # simulation: truth drawn from the M3 or M1 posterior (node -> planted hands -> active decisions)
    sel = np.isin(hid_all, hh); dh = np.searchsorted(hh, hid_all[sel]); rr = r_all[sel]; pre = pre_all[sel]
    for tr_ in ("M3", "M1"):
        qp2, w2, keep, u, rho = cache[tr_]; th = post[tr_]["th"]; uu = np.repeat(u, len(rho))[keep]
        for _ in range(NS):
            j = rng.choice(len(w2), p=w2); planted = rng.random(len(hs)) < qp2[:, j]
            s = expit(np.where(pre, th[0], th[1]) + uu[j]); pact = s * rr / ((1 - s) + s * rr)
            act_d = (rng.random(len(rr)) < pact) & planted[dh]; act_h = np.zeros(len(hs), bool); np.logical_or.at(act_h, dh, act_d)
            wit_h = np.zeros(len(hs), bool)
            if WIT: chg_d = act_d & (rng.random(len(rr)) < np.maximum(1 - 1 / np.maximum(rr, 1e-12), 0)); np.logical_or.at(wit_h, dh, chg_d)
            for rule, flag in (("ACT", act_h & (pw > 0)), ("PLANT", planted & (pw > 0)), ("WITNESS", wit_h & (pw > 0)))[:len(RULES)]:
                tru = set(hs[np.flatnonzero(flag)[:5]])
                if not tru: continue
                for k in names:
                    hit = 0; sc = 0.0
                    for i, h in enumerate(picks[k][pid]):
                        if h in tru: hit += 1; sc += hit / (i + 1)
                    sim[(tr_, rule, k)].append(sc / min(5, len(tru)))
for tr_ in ("M3", "M1"):
    for rule in RULES:
        print(f"SIM truth={tr_} rule={rule}: " + ", ".join(f"{k} {np.mean(sim[(tr_, rule, k)]):.4f}" for k in names), flush=True)
ref = pd.read_csv(f"{C}/r6_re_subh.csv", dtype=str).set_index("pair_id")
print("m1_act identical to r6_re_subh:", sum(all(hi2id[x] == y for x, y in zip(picks["m1_act"][p], ref.loc[p, EVC])) for p in oth), "/", len(oth), "(M1 here refits s too)")
for a, b in (("fix_act", "m3_act"), ("fix_plant", "m3_plant"), ("m1_act", "m3_act"), ("m1_plant", "m3_plant")):
    ov = [len(set(picks[a][p]) & set(picks[b][p])) for p in oth]; print(f"overlap {a} vs {b}: mean {np.mean(ov):.2f}, pairs changed {sum(o < 5 for o in ov)}")
pd.DataFrame(info).to_parquet(f"{R3}/t37_re2d_pairs_{PREFIX}.parquet")
json.dump(dict(full={k: v.tolist() for k, v in full.items()}, fits=FITLOG, cv=cv, sim={f"{a}|{b}|{c}": float(np.mean(v)) for (a, b, c), v in sim.items()}), open(f"{R3}/t37_re2d_{PREFIX}.json", "w"), indent=1)
for k, name in OUTS:
    out = bidx.copy(); out.loc[oth, EVC] = [[hi2id[x] for x in picks[k][p]] for p in oth]
    o = out.reset_index()[base.columns]; path = f"{C}/{PREFIX}_{name}.csv"; o.to_csv(path, index=False)
    rec = dict(file=os.path.basename(path), base=BASE, f4ev=k, env={e: os.environ.get(e) for e in ("PREFIX", "NR", "NU", "NS", "OOT", "RV1", "WIT", "SKIPCV")}, m3=full["M3"].tolist(), fits_ok=all(f["success"] for f in FITLOG), source_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(), sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(),
               evidence_rows_changed_vs_base=int((o[EVC].values != base[EVC].values).any(1).sum()),
               risk_or_behavior_changed=int(((o.risk_score.values != base.risk_score.values) | (o.predicted_behavior.values != base.predicted_behavior.values)).sum()))
    json.dump(rec, open(path.replace(".csv", ".receipt.json"), "w"), indent=1); print(json.dumps(rec))
