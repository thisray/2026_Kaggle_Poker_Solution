"""R3-P30: where do the real hidden positives stop? Calibrating the PU threshold with the persistence discriminator.

The `hid` set is "reference-model OOF > 0.3", a threshold nobody derived. t88 showed that labelled positives persist
across the two independent 2/3 dev realisations at 0.958 while a noise-driven high score would not. So sweep the
threshold and watch persistence: while the flagged pairs are real collusion it stays near the labelled-positive level;
where it collapses towards the unlabelled background, the real positives have run out.
The eval-phase check (real collusion stops in eval) is reported alongside as a second, independent axis.
"""
import numpy as np, pandas as pd, json
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
pidx = pd.read_parquet(f"{O}/np/player_index.parquet"); pm = dict(zip(pidx.player_id, pidx.pi))
lab = pd.read_csv(f"{RAW}/development_labels.csv")
lab["key"] = np.minimum(lab.player_1.map(pm), lab.player_2.map(pm)) * 12000 + np.maximum(lab.player_1.map(pm), lab.player_2.map(pm))
pos = set(lab.key[lab.label == 1])
REF = "m15_o_pos_a"
d = pd.read_parquet(f"{O}/{REF}_train_oof.parquet")
P = d[d.src.isin(["devsub11", "devsub12"])].pivot_table(index="key", columns="src", values="oof").dropna()
lbl = P.index.isin(pos)
es = pd.read_parquet(f"{O}/{REF}_eval_scores.parquet")
es["key"] = es.p_lo * 12000 + es.p_hi; es["pct"] = es.groupby("pool").score.rank(pct=True)
EP = es.set_index("key").pct
print(f"reference model {REF}; pairs scored in both subsamples {len(P)}; labelled positives {int(lbl.sum())}")
base = float((P.devsub11[lbl] > 0.3).mean())
p_lab = float(((P.devsub11 > 0.3) & (P.devsub12 > 0.3))[lbl].sum() / max((P.devsub11 > 0.3)[lbl].sum(), 1))
print(f"labelled positives: persistence at 0.3 = {p_lab:.3f}\n")
rows = []
for t in (0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005):
    a = (P.devsub11 > t) & ~lbl; b = (P.devsub12 > t) & ~lbl
    both = int((a & b).sum()); n = int(a.sum())
    persist = both / max(n, 1)
    # eval-phase percentile of the pairs flagged in both subsamples (real collusion should look ordinary there)
    ks = P.index[(a & b).values]
    ev = EP.reindex(ks).dropna()
    rows.append(dict(thr=t, flagged_sub11=n, flagged_both=both, persistence=round(persist, 3),
                     eval_pct_median=round(float(ev.median()), 3) if len(ev) else None,
                     eval_above_p99=round(float((ev > 0.99).mean()), 3) if len(ev) else None, n_eval=len(ev)))
    print(json.dumps(rows[-1]), flush=True)
# a random-score control at the same counts
rs = np.random.default_rng(3); ctrl = []
for t in (0.3, 0.05, 0.01):
    n = int(((P.devsub11 > t) & ~lbl).sum())
    idx = rs.choice(np.flatnonzero(~lbl), n, replace=False)
    sec = P.devsub12.values[idx] > t
    ctrl.append(dict(thr=t, n=n, persistence_if_independent=round(float(sec.mean()), 3)))
print("\nindependence control (same counts, random pairs):", json.dumps(ctrl))
json.dump(dict(sweep=rows, control=ctrl, labelled_persistence=p_lab), open(f"{O}/r3/t98_hid_threshold.json", "w"), indent=1)
