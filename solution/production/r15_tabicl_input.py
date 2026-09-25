"""Join the historical 11 candidate scores with 13 gameplay extras for TabICLv2."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCORES = [
    "sc_r5", "u0", "u_rr", "u_r5b", "lin_contrib", "nn_contrib",
    "t1_score", "s1_stage1", "gen_logit", "gen_rank_pct", "rank_u_r5b",
]
EXTRAS = [
    "o_lost_dr", "o_flow_dr", "tpl_cos", "wit_r2c_max", "o_dir_agree",
    "facing_mx", "S_sur_aggr_act_mx", "tpl_mass", "eq_fold_to_mx",
    "P_pos_mx", "o_net", "flow_mx", "z_o_contrib",
]


def build(candidate_path: Path, extras_path: Path, output_path: Path) -> None:
    candidates = pd.read_csv(candidate_path)
    extras = pd.read_csv(extras_path)
    keys = ["slot", "hand_id"]
    for name, frame in (("candidates", candidates), ("extras", extras)):
        if frame.duplicated(keys).any():
            raise ValueError(f"Duplicate {name} keys")
    missing = set(keys + SCORES) - set(candidates)
    missing |= set(keys + EXTRAS) - set(extras)
    if missing:
        raise ValueError(f"Missing TabICL input fields: {sorted(missing)}")
    metadata = [name for name in ("pair_id", "pool", "fold", "ev", "m_p") if name in candidates]
    result = candidates[keys + metadata + SCORES].merge(
        extras[keys + EXTRAS], on=keys, how="left", validate="one_to_one"
    )
    if result[EXTRAS].isna().any().any():
        raise ValueError("Gameplay extras do not cover every candidate")
    result.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--extras", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.candidates, args.extras, args.out)
