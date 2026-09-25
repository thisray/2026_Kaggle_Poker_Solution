"""Round-3 research probes (read-only on existing GB10 artifacts).

Usage (GB10, kaggle_poker_opus_260917 python):
    OMP_NUM_THREADS=2 nice -n 15 python scripts/round3_probes.py ring
    OMP_NUM_THREADS=2 nice -n 15 python scripts/round3_probes.py trio
    OMP_NUM_THREADS=2 nice -n 15 python scripts/round3_probes.py decoder

Artifacts expected under ART (opus_r1_20260917). Output JSON goes to OUT.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import poisson

ART = os.environ.get(
    "ART", "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
)
RAW = os.environ.get(
    "RAW", "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
)
OUT = os.environ.get(
    "OUT", "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917"
)


def load_oof(name="m25t1_handfeat2_m19w10_oof"):
    return pd.read_parquet(f"{ART}/{name}.parquet").sort_values(["sl", "ts"]).reset_index(drop=True)


def map5_rows(df, col):
    aps = []
    for _, g in df.groupby("sl"):
        rel = set(g.h[g.ev])
        g = g.sort_values(col, ascending=False)
        hits = 0
        s = 0.0
        for i, hh in enumerate(g.h.values[:5]):
            if hh in rel:
                hits += 1
                s += hits / (i + 1)
        aps.append(s / min(5, max(len(rel), 1)))
    return float(np.mean(aps))


def probe_ring():
    labels = pd.read_csv(f"{RAW}/development_labels.csv")
    v7 = pd.read_parquet(f"{ART}/m15_v7_drop_contrast_train_oof.parquet")
    ev = pd.read_parquet(f"{ART}/m15_v7_drop_contrast_eval_scores.parquet")
    v7["pid_lo"] = v7.key // 12000
    v7["pid_hi"] = v7.key % 12000
    hid = v7[(v7.label == -1) & (v7.oof > 0.3)].drop_duplicates("key")
    hidden_players = set(hid.pid_lo) | set(hid.pid_hi)
    lab = labels[labels.label == 1]
    report = {
        "dev_hidden_pairs_oof_gt_0.3": int(len(hid)),
        "dev_hidden_players": int(len(hidden_players)),
        "labeled_positive_pairs": int(len(lab)),
    }
    ehi = ev[ev.score > 0.5].copy()
    ehi["pid_lo"] = ehi.key // 12000
    ehi["pid_hi"] = ehi.key % 12000
    n = len(ehi)
    ph = len(hidden_players) / 12000.0
    report["eval_high_pairs_gt_0.5"] = int(n)
    report["eval_high_touching_dev_hidden"] = int(
        (ehi.pid_lo.isin(hidden_players) | ehi.pid_hi.isin(hidden_players)).sum()
    )
    report["eval_high_touching_dev_hidden_expected_indep"] = round(
        float(2 * n * ph - n * ph * ph), 1
    )
    counts = pd.concat([ehi.pid_lo, ehi.pid_hi]).value_counts()
    report["eval_high_players_in_ge2_pairs"] = int((counts >= 2).sum())
    with open(f"{OUT}/r3_probe_playerring.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def probe_trio():
    labels = pd.read_csv(f"{RAW}/development_labels.csv")
    pos = labels[labels.label == 1]
    partners = {}
    for _, r in pos.iterrows():
        partners.setdefault(r.player_1, set()).add(r.player_2)
        partners.setdefault(r.player_2, set()).add(r.player_1)
    multi = {p: ps for p, ps in partners.items() if len(ps) >= 2}
    pairs = set(tuple(sorted((r.player_1, r.player_2))) for _, r in pos.iterrows())
    closed = 0
    for p, ps in multi.items():
        ps = sorted(ps)
        if any(tuple(sorted((a, b))) in pairs for i, a in enumerate(ps) for b in ps[i + 1:]):
            closed += 1
    report = {
        "players_with_multiple_partners": int(len(multi)),
        "degree_distribution": {
            str(k): int(sum(1 for ps in multi.values() if len(ps) == k))
            for k in sorted({len(ps) for ps in multi.values()})
        },
        "closed_triangles": int(closed),
    }
    with open(f"{OUT}/r3_probe_trio.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def probe_decoder():
    D = load_oof()
    S = D.sc_fam.values
    cs = D.groupby("sl").sc_fam.cumsum().values - S
    tp = D.groupby("sl").ts.rank(pct=True).values
    folds = (D.groupby("sl").ngroup() % 5).values
    res = {f"K{k}_a{a}": S * poisson.cdf(k, cs) * np.exp(-a * tp)
           for k in [1, 2, 3, 4, 5, 6] for a in [0.0, 0.25, 0.5]}
    out = {name: round(map5_rows(D.assign(y=y), "y"), 4) for name, y in res.items()}
    nested = []
    for f in range(5):
        tr, va = folds != f, folds == f
        Dt, Dv = D[tr].copy(), D[va].copy()
        best, bestv = None, -1.0
        for c in res:
            Dt["y"] = res[c][tr]
            v = map5_rows(Dt, "y")
            if v > bestv:
                bestv, best = v, c
        Dv = Dv.copy()
        Dv["y"] = res[best][va]
        nested.append(map5_rows(Dv, "y"))
    out["nested_mean"] = round(float(np.mean(nested)), 4)
    out["nested_folds"] = [round(x, 4) for x in nested]
    out["baseline_K4_a0"] = round(map5_rows(D.assign(y=res["K4_a0.0"]), "y"), 4)
    out["best_fixed_K3_a0.25"] = round(map5_rows(D.assign(y=res["K3_a0.25"]), "y"), 4)
    with open(f"{OUT}/r3_probe_decoder5.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", choices=["ring", "trio", "decoder"])
    args = ap.parse_args()
    {"ring": probe_ring, "trio": probe_trio, "decoder": probe_decoder}[args.probe]()
