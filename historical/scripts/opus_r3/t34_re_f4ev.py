"""R3-F13: pair-specific planting rate (R19 Beta random effects) for fourth-family evidence.

For the other_coordination pairs of a base candidate:
  1. rebuild the full co-seated hand sequence (all shared hands, pair-win flag, lf/l0 from t17; 0 for hands without decisions),
  2. fit Beta(mu*kappa, (1-mu)*kappa) hyperparameters of the pair planting rate rho_p (R19 f4_random_effects),
  3. decode first-5 events at every quadrature node of rho_p and integrate afterwards (E[DP(q)], not DP(E[q])),
  4. regression-check the fixed-rho decoders against the r5 candidates,
  5. self-consistent simulation: truth drawn from the RE posterior (and from the fixed-rho posterior) under the ACT and
     PLANT labeller rules, scoring fixed-rho and RE pickers,
  6. write E-only candidates (risk, behaviour and non-F4 evidence untouched).
"""
import os, sys, json, hashlib
import numpy as np, pandas as pd
from scipy.special import expit, logit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pairindex as PI
from f4_random_effects import quadrature, fit as re_fit, score as re_score, pair_tables, first_k_at_nodes

A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"; R3 = f"{OUT}/r3"
BASE = os.environ.get("BASE", "r5_subh.csv"); PREFIX = os.environ.get("PREFIX", "r6_re"); NS = int(os.environ.get("NS", "300")); NODES = 160
RHO0, SPRE, SPOST = 0.70919, 0.50769, 0.42580          # R3 hierarchical fit (t17)
EVC = [f"evidence_hand_{i}" for i in range(1, 6)]
base = pd.read_csv(f"{C}/{BASE}", dtype=str); bidx = base.set_index("pair_id")
oth = bidx.index[bidx.predicted_behavior == "other_coordination"].tolist()
e85 = pd.read_parquet(f"{OUT}/s85_eval_bf.parquet")[["slot", "pair_id"]].set_index("pair_id").slot
slots = e85.loc[oth].values; sl2pid = dict(zip(slots, oth))
hq = pd.read_parquet(f"{R3}/t17_hands_eval.parquet")[["slot", "h", "lf", "l0", "q_act", "q_plant"]]; hq = hq[hq.slot.isin(slots)]
H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, slots); H, S, T, SL = H[m], S[m], T[m], SL[m]
won = np.load(f"{D}/s_won.npy", mmap_mode="r"); ts = np.load(f"{D}/h_ts.npy"); W = np.asarray(won[H]); ix = np.arange(len(H))
G = pd.DataFrame({"slot": SL, "h": H, "ts": ts[H], "pw": (W[ix, S] > 0) | (W[ix, T] > 0)}).merge(hq, on=["slot", "h"], how="left")
G["dec"] = G.lf.notna(); G = G.fillna({"lf": 0.0, "l0": 0.0, "q_act": 0.0, "q_plant": 0.709}).sort_values(["slot", "ts", "h"]).reset_index(drop=True)
assert G.slot.nunique() == len(oth), "every other pair must have co-seated hands"
print(f"base {BASE}: {len(oth)} other pairs, {len(G)} co-seated hands, {int(G.dec.sum())} with member decisions, pair-win rate {G.pw.mean():.3f}")

# hyperparameters: current other pairs (all hands; hands without decisions contribute exactly 0 to the likelihood)
d = G[["slot", "lf"]].copy()
par = re_fit(d, NODES); mu, kappa = float(expit(par[0])), float(np.exp(par[1]))
ll_fixed = float(np.logaddexp(np.log1p(-RHO0), np.log(RHO0) + d.lf.values).sum())
print(f"RE fit on current pairs: mu {mu:.4f} kappa {kappa:.4f} LLR {re_score(par, d, NODES):.3f} vs fixed rho {RHO0} LLR {ll_fixed:.3f}")
rho, lw = quadrature(mu, kappa, NODES); z = logit(rho)

def first_k_prob(q, K=5):
    o = np.zeros(len(q)); dist = np.zeros(K + 1); dist[0] = 1.0
    for i, p in enumerate(q):
        o[i] = p * dist[:K].sum(); nd = dist * (1 - p); nd[1:] += dist[:-1] * p; nd[K] += dist[K] * p; dist = nd
    return o

def top5(score, hs):
    p = hs[np.argsort(-score, kind="stable")][:5]; assert len(set(p)) == 5; return p

hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hi2id = dict(zip(hidx.hi, hidx.hand_id)); id2hi = {v: k for k, v in hi2id.items()}
rows = pd.read_parquet(f"{R3}/t17_rows_eval.parquet", columns=["h", "st", "slot", "r"]); rows = rows[rows.slot.isin(slots)]
ref = {k: pd.read_csv(f"{C}/{v}", dtype=str).set_index("pair_id") for k, v in (("subh", "r5_subh.csv"), ("subp", "r5_subp.csv"))}
picks = {k: {} for k in ("fix_act", "fix_plant", "re_act", "re_plant", "re_mix")}; info = []; rng = np.random.default_rng(20260919)
sim = {(truth, rule, k): [] for truth in ("RE", "FIX") for rule in ("ACT", "PLANT") for k in picks}
for sl, g in G.groupby("slot", sort=False):
    pid = sl2pid[sl]; hs = g.h.values; pw = g.pw.values.astype(float); lf = g.lf.values; l0 = g.l0.values
    # fixed-rho decoders exactly as t21 (regression check against r5)
    fa = first_k_prob(g.q_act.values * pw) + 1e-6 * first_k_prob(g.q_act.values)
    fp = first_k_prob(g.q_plant.values * pw) + 1e-6 * first_k_prob(g.q_plant.values)
    picks["fix_act"][pid] = top5(fa, hs); picks["fix_plant"][pid] = top5(fp, hs)
    # random-effects decoders: posterior over rho_p, DP at every node, integrate afterwards
    llp = np.logaddexp(np.log1p(-rho)[None, :], np.log(rho)[None, :] + lf[:, None]).sum(0); wp = np.exp(lw + llp - (lw + llp).max()); wp /= wp.sum()
    qp = expit(z[None, :] + lf[:, None]); qa = qp * (-np.expm1(np.minimum(l0 - lf, 0)))[:, None]
    ra = first_k_at_nodes(qa * pw[:, None]) @ wp + 1e-6 * (first_k_at_nodes(qa) @ wp)
    rp = first_k_at_nodes(qp * pw[:, None]) @ wp + 1e-6 * (first_k_at_nodes(qp) @ wp)
    picks["re_act"][pid] = top5(ra, hs); picks["re_plant"][pid] = top5(rp, hs); picks["re_mix"][pid] = top5(ra + rp, hs)
    rm = float(rho @ wp); info.append(dict(pair_id=pid, slot=int(sl), hands=len(hs), dec_hands=int(g.dec.sum()), pw_rate=float(pw.mean()), rho_mean=rm,
                                           ov_act=len(set(picks["fix_act"][pid]) & set(picks["re_act"][pid])), ov_plant=len(set(picks["fix_plant"][pid]) & set(picks["re_plant"][pid]))))
    # self-consistent simulation
    pos = {h: i for i, h in enumerate(hs)}; r = rows[rows.slot == sl]; hi_ = r.h.map(pos).values; ok = ~np.isnan(hi_.astype(float))
    hi_ = hi_[ok].astype(int); sd = np.where(r.st.values[ok] == 0, SPRE, SPOST); rr = r.r.values[ok]
    pact = sd * rr / ((1 - sd) + sd * rr)
    qfix = expit(logit(RHO0) + lf)
    for truth in ("RE", "FIX"):
        for _ in range(NS):
            q_ = qp[:, rng.choice(len(rho), p=wp)] if truth == "RE" else qfix
            planted = rng.random(len(hs)) < q_
            act_d = (rng.random(len(rr)) < pact) & planted[hi_]
            act_h = np.zeros(len(hs), bool); np.logical_or.at(act_h, hi_, act_d)
            for rule, flag in (("ACT", act_h & (pw > 0)), ("PLANT", planted & (pw > 0))):
                tr = set(hs[np.flatnonzero(flag)[:5]])
                if not tr: continue
                for k in picks:
                    hit = 0; sc = 0.0
                    for i, h in enumerate(picks[k][pid]):
                        if h in tr: hit += 1; sc += hit / (i + 1)
                    sim[(truth, rule, k)].append(sc / min(5, len(tr)))
info = pd.DataFrame(info)
for k, rk in (("fix_act", "subh"), ("fix_plant", "subp")):
    same = sum(all(hi2id[x] == y for x, y in zip(picks[k][p], ref[rk].loc[p, EVC])) for p in oth)
    print(f"regression {k} vs r5_{rk}: {same}/{len(oth)} pairs identical")
print(f"rho_mean quantiles {np.round(info.rho_mean.quantile([0, .1, .25, .5, .75, .9, 1]).values, 3).tolist()}")
print(f"overlap fixed vs RE (of 5): ACT mean {info.ov_act.mean():.2f} (pairs changed {int((info.ov_act < 5).sum())}), PLANT mean {info.ov_plant.mean():.2f} (pairs changed {int((info.ov_plant < 5).sum())})")
for truth in ("RE", "FIX"):
    for rule in ("ACT", "PLANT"):
        print(f"SIM truth={truth} rule={rule}: " + ", ".join(f"{k} {np.mean(sim[(truth, rule, k)]):.4f}" for k in picks))
info.to_parquet(f"{R3}/t34_re_pairs.parquet"); json.dump(dict(mu=mu, kappa=kappa, n_pairs=len(oth)), open(f"{R3}/t34_re_fit.json", "w"))
for k, name in (("re_act", "subh"), ("re_plant", "subp"), ("re_mix", "mix")):
    out = bidx.copy(); out.loc[oth, EVC] = [[hi2id[x] for x in picks[k][p]] for p in oth]
    o = out.reset_index()[base.columns]; path = f"{C}/{PREFIX}_{name}.csv"; o.to_csv(path, index=False)
    rec = dict(file=os.path.basename(path), base=BASE, f4ev=k, mu=mu, kappa=kappa, sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(),
               evidence_rows_changed_vs_base=int((o[EVC].values != base[EVC].values).any(1).sum()),
               risk_or_behavior_changed=int(((o.risk_score.values != base.risk_score.values) | (o.predicted_behavior.values != base.predicted_behavior.values)).sum()))
    json.dump(rec, open(path.replace(".csv", ".receipt.json"), "w"), indent=1); print(json.dumps(rec))
