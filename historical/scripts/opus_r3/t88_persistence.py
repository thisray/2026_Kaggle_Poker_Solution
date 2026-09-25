"""R3-P25: do the suspected hidden positives behave like real collusion across independent dev realisations?

devsub11 and devsub12 are two different 2/3 realisations of the same dev phase. A real colluding pair keeps ~2/3 of
its ~6 planted hands in each, so it should score high in BOTH. A pair that is high only because a few particular hands
happened to look bad should be high in one and ordinary in the other.
Persistence = P(high in the other subsample | high in this one), computed identically for labelled positives (the
calibration group, known real) and for the unlabelled suspects.
"""
import numpy as np, pandas as pd
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
pidx = pd.read_parquet(f"{O}/np/player_index.parquet"); pm = dict(zip(pidx.player_id, pidx.pi))
lab = pd.read_csv(f"{RAW}/development_labels.csv")
lab["key"] = np.minimum(lab.player_1.map(pm), lab.player_2.map(pm)) * 12000 + np.maximum(lab.player_1.map(pm), lab.player_2.map(pm))
pos = set(lab.key[lab.label == 1])
for TAG in ("m5_both", "m15_o_pos_a"):
    d = pd.read_parquet(f"{O}/{TAG}_train_oof.parquet")
    P = d[d.src.isin(["devsub11", "devsub12"])].pivot_table(index="key", columns="src", values="oof")
    P = P.dropna(); lbl = P.index.isin(pos)
    r11 = P.devsub11.rank(pct=True); r12 = P.devsub12.rank(pct=True)
    print(f"\n{TAG}: pairs in both subsamples {len(P)}  (labelled positives {int(lbl.sum())})")
    for thr, nm in ((0.3, "score>0.3"), (None, "top-400 by rank")):
        if thr is not None: h11 = P.devsub11 > thr; h12 = P.devsub12 > thr
        else: h11 = r11 > 1 - 400 / len(P); h12 = r12 > 1 - 400 / len(P)
        for grp, m in (("labelled positives", lbl), ("unlabelled", ~lbl)):
            a = h11 & m; b = h12 & m
            if a.sum() == 0: continue
            print(f"   {nm:16s} {grp:20s}: high in sub11 {int(a.sum())}, also high in sub12 {int((a & h12).sum())} "
                  f"({(a & h12).sum() / max(a.sum(), 1):.3f}); symmetric persistence {(a & h12).sum() * 2 / max(a.sum() + b.sum(), 1):.3f}")
