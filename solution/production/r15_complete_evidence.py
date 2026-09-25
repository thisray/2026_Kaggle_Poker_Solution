"""Gate R5 candidates and apply scored known-family evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EVIDENCE = [f"evidence_hand_{rank}" for rank in range(1, 6)]
def gated_pairs(base_path: Path, budget: int) -> set[str]:
    base = pd.read_csv(base_path, dtype={"pair_id": str})
    ranked = base.sort_values(["risk_score", "pair_id"], ascending=[False, True])
    return set(ranked.head(budget).pair_id)


def prepare(
    candidates_path: Path,
    r10_path: Path,
    r32_path: Path,
    output_dir: Path,
    budget: int,
) -> None:
    pairs = gated_pairs(r10_path, budget) | gated_pairs(r32_path, budget)
    candidates = pd.read_parquet(candidates_path)
    candidates = candidates[candidates.pair_id.astype(str).isin(pairs)].copy()
    if candidates.empty or candidates.duplicated(["slot", "hand_id"]).any():
        raise ValueError("Missing or duplicate gated candidates")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(output_dir / "gated_candidates.parquet", index=False)
    print(json.dumps({"gated_pairs": len(pairs), "candidate_rows": len(candidates)}))


def patch(base_path: Path, scored_path: Path,
          output_path: Path, budget: int) -> None:
    base = pd.read_csv(base_path, dtype={"pair_id": str})
    scored = pd.read_csv(scored_path, dtype={"pair_id": str, "hand_id": str})
    keep = gated_pairs(base_path, budget)
    routed = base.set_index("pair_id").predicted_behavior
    scored = scored[scored.pair_id.isin(keep)].copy()
    scored["family"] = scored.pair_id.map(routed)
    scored = scored[scored.family.ne("other_coordination")].copy()
    scored = scored.sort_values(
        ["pair_id", "score", "hand_id"], ascending=[True, False, True], kind="stable"
    )
    top = scored.groupby("pair_id", sort=False).head(5).copy()
    full = top.groupby("pair_id").size().loc[lambda counts: counts.eq(5)].index
    top = top[top.pair_id.isin(full)]
    if top.duplicated(["pair_id", "hand_id"]).any():
        raise ValueError("Evidence contains duplicate pair-hand keys")
    wide = top.assign(rank=top.groupby("pair_id").cumcount() + 1).pivot(
        index="pair_id", columns="rank", values="hand_id"
    )
    output = base.set_index("pair_id").copy()
    for rank, column in enumerate(EVIDENCE, start=1):
        output.loc[wide.index, column] = wide[rank].to_numpy()
    output = output.reset_index()[base.columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(json.dumps({"output": str(output_path), "patched_pairs": len(wide)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepared = commands.add_parser("prepare")
    for name in ("candidates", "r10", "r32", "out-dir"):
        prepared.add_argument("--" + name, type=Path, required=True)
    prepared.add_argument("--budget", type=int, default=4000)
    patched = commands.add_parser("patch")
    for name in ("base", "scored", "out"):
        patched.add_argument("--" + name, type=Path, required=True)
    patched.add_argument("--budget", type=int, default=4000)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.candidates, args.r10, args.r32, args.out_dir, args.budget)
    else:
        patch(args.base, args.scored, args.out, args.budget)


if __name__ == "__main__":
    main()
