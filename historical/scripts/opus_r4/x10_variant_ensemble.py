"""R4-X10: average the OOF event probabilities of the feature variants (A: flow orientation + t58 block, B: candidate-flow orientation + t58 block,
C: candidate-flow orientation + corrected block) before the DP. DT scored by rank average, CI by the probability stack; paired pool bootstrap vs variant A and vs R15."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18")
import ci_censored_event as C
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; TAGS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["x4", "x4c", "x4f"]
hmap = pd.read_parquet(f"{O}/np/hand_index.parquet").set_index("hand_id").hi; t45 = pd.read_parquet(f"{O}/r3/t45_known_e_rerank.parquet"); t45["h"] = t45.hand_id.map(hmap)
cand = pd.read_parquet(f"{O}/t4_wrong_vs_hit.parquet").rename(columns={"sl": "slot"}); lg = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))
for fam2 in ("di", "co", "so"):
    rows = [pd.read_parquet(f"{O}/r4/{t}_rows_{fam2}.parquet").reset_index(drop=True) for t in TAGS]; s = rows[0]
    for r in rows[1:]: assert (r.slot.values == s.slot.values).all() and (r.h.values == s.h.values).all()
    P = {t: np.load(f"{O}/r4/{t}_oofp_{fam2}.npy") for t in TAGS}; P["ENS"] = np.mean([P[t] for t in TAGS], axis=0); counts = s.groupby("slot").ev.sum()
    c = cand[cand.slot.isin(s.slot)].drop(columns=["ev", "ts"], errors="ignore").merge(s[["slot", "h", "ts", "ev"]], on=["slot", "h"], validate="one_to_one").merge(t45[["slot", "h", "tab"]], on=["slot", "h"], how="left").reset_index(drop=True)
    mi = pd.MultiIndex.from_arrays([c.slot, c.h]); idx = counts.index; pl = pd.Series(idx // 900, index=idx); up = pl.unique(); rng = np.random.RandomState(8)
    boot = lambda d: float(np.mean([np.mean(np.concatenate([pd.Series(d, index=idx)[pl == p_].values for p_ in rng.choice(up, len(up))])) > 0 for _ in range(1000)]))
    base = C.pair_ap(c, -c.r, counts).values; out = {}
    for name, PP in P.items():
        for mode, par in (("rank", 0.25), ("rank", 0.35), ("rank", 0.5), ("stack", 1.0), ("stack", 3.0)):
            v = []
            for p in PP:
                q = C.first_k_marginal(s, p)
                if mode == "rank": j = C.rank_candidates(s, c.drop(columns=["q", "newscore"], errors="ignore"), q, weight=par); v.append(C.pair_ap(j, j.newscore, counts).values)
                else: qc = pd.Series(q, index=pd.MultiIndex.from_arrays([s.slot, s.h])).reindex(mi).values; v.append(C.pair_ap(c, lg(c.tab.values) + par * lg(qc), counts).values)
            out[(name, mode, par)] = np.mean(v, axis=0)
    print(f"== {fam2}: R15 {base.mean():.4f}")
    for mode, par in (("rank", 0.25), ("rank", 0.35), ("rank", 0.5), ("stack", 1.0), ("stack", 3.0)):
        print(f"   {mode} {par}: " + "  ".join(f"{n} {out[(n, mode, par)].mean():.4f}" for n in P) + f" | ENS-R15 P>0 {boot(out[('ENS', mode, par)] - base):.3f} | ENS-{TAGS[0]} {out[('ENS', mode, par)].mean() - out[(TAGS[0], mode, par)].mean():+.4f} P>0 {boot(out[('ENS', mode, par)] - out[(TAGS[0], mode, par)]):.3f}")
