"""Small, read-only CI OOF oracle decomposition; never constructs a submission."""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    cp = args.root / "ci_oof.candidates.csv"
    jp = args.root / "ci_oof.json"
    meta = json.loads(jp.read_text())
    # Every public evidence list has at most five elements. Equality establishes
    # denominator five for each of the reported CI queries, including missed items.
    assert meta["n_evidence"] == 5 * meta["n_pairs"]
    d = pd.read_csv(cp, usecols=["slot", "h", "ts", "ev", "r", "pa_at_trig", "y1", "newscore"])
    assert d.slot.nunique() == meta["n_pairs"]
    assert not d.duplicated(["slot", "h"]).any()
    assert d.ev.isin([0, 1, False, True]).all()
    d["hard_score"] = d.newscore - 100 * (~((d.pa_at_trig == 6) & d.y1.isin([2, 3])))
    d["base_score"] = -d.r
    results = []
    strata = []
    for slot, group in d.groupby("slot"):
        for name, col in [("r15", "base_score"), ("r18", "newscore"), ("r18hard", "hard_score")]:
            g = group.sort_values([col, "ts", "h"], ascending=[False, True, True], kind="stable")
            y = g.ev.astype(int).to_numpy()
            h5 = int(y[:5].sum())
            ap = float((y[:5] * np.cumsum(y[:5]) / np.arange(1, 6)).sum() / 5)
            results.append({"variant": name, "ap5": ap,
                            "candidate_denominator_ap5": ap * 5 / min(5, int(y.sum())) if y.sum() else 0.0,
                            "fixed_set_order_oracle": h5 / 5,
                            "top10_selection_oracle": min(5, int(y[:10].sum())) / 5,
                            "top20_selection_oracle": min(5, int(y[:20].sum())) / 5,
                            "hits_top5": h5, "miss_rank6_10": int(y[5:10].sum()),
                            "miss_rank11_20": int(y[10:20].sum()),
                            "missing_from20": 5 - int(y.sum()),
                            "hard_pass_oracle": min(5, int(g.loc[(g.pa_at_trig == 6) & g.y1.isin([2, 3]), "ev"].sum())) / 5,
                            "pool": int(slot // 900)})
            if name == "r18hard":
                for label, z in [("false_top5", g.head(5).query("ev == 0")),
                                 ("missed_truth", g.iloc[5:].query("ev == 1"))]:
                    for _, row in z.iterrows():
                        strata.append({"kind": label, "first_member_action": int(row.y1),
                                       "active_at_trigger": int(row.pa_at_trig)})
    r = pd.DataFrame(results)
    assert abs(r.loc[r.variant == "r18", "ap5"].mean() - meta["R18_CI_E"]) < 1e-10
    summary = {}
    for name, g in r.groupby("variant"):
        summary[name] = {c: float(g[c].mean()) for c in ["ap5", "candidate_denominator_ap5", "fixed_set_order_oracle", "top10_selection_oracle", "top20_selection_oracle", "hard_pass_oracle"]}
        summary[name].update({c: int(g[c].sum()) for c in ["hits_top5", "miss_rank6_10", "miss_rank11_20", "missing_from20"]})
        summary[name]["pools"] = int(g.pool.nunique())
    print(json.dumps({"observed_at": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
                      "python": platform.python_version(), "pairs": int(meta["n_pairs"]),
                      "input_manifest": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in (cp, jp)},
                      "summary": summary, "error_cells": pd.DataFrame(strata).value_counts().rename("count").reset_index().to_dict("records"),
                      "limitation": "Exploratory frozen-policy development OOF; oracle is not deployable gain; no hidden labels or retraining."}, indent=2))


if __name__ == "__main__":
    main()
