"""Symmetric, unit-aware reference features. No IDs or label artifacts."""
from __future__ import annotations
import math
from collections.abc import Mapping
import numpy as np


def pair_outcome_features(net_a: float, net_b: float, contribution_a: float,
                          contribution_b: float, big_blind: float) -> dict[str, float]:
    vals = (net_a, net_b, contribution_a, contribution_b, big_blind)
    if not all(math.isfinite(float(x)) for x in vals) or big_blind <= 0:
        raise ValueError("Finite outcomes and strictly positive big_blind required")
    if contribution_a < 0 or contribution_b < 0:
        raise ValueError("Contribution cannot be negative")
    a, b, ca, cb = [x / big_blind for x in vals[:4]]
    ab = min(max(-a, 0), max(b, 0))
    ba = min(max(-b, 0), max(a, 0))
    return {"f_pair_net_bb": a+b, "f_abs_net_gap_bb": abs(a-b),
            "f_transfer_proxy_max_bb": max(ab, ba),
            "f_transfer_proxy_sum_bb": ab+ba,
            "f_contribution_sum_bb": ca+cb,
            "f_contribution_abs_gap_bb": abs(ca-cb)}


def is_price_increasing(action: str, incremental_amount: float,
                        to_call: float, tolerance: float = 1e-9) -> bool:
    """Whether an action increases price, NOT whether betting is reopened.

    incremental_amount must already have been normalized by a verified adapter.
    An under-raise all-in may increase price without reopening betting.
    """
    if not all(math.isfinite(x) for x in (incremental_amount, to_call)):
        raise ValueError("Amounts must be finite")
    if min(incremental_amount, to_call) < 0:
        raise ValueError("Negative increment or call cost")
    a = str(action).lower().replace("-", "_").replace(" ", "_")
    if a in {"fold", "check", "call", "post_sb", "post_bb", "ante"}:
        return False
    if a in {"bet", "raise", "all_in", "allin"}:
        return incremental_amount > to_call + tolerance
    raise ValueError(f"Unknown action: {action}; update and test the schema adapter")


def standardized_binary_residual(observed: list[int], expected: list[float],
                                 variance_floor: float = 1.0) -> float:
    y, p = np.asarray(observed), np.asarray(expected, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError("Equal-length binary observations required")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or variance_floor <= 0:
        raise ValueError("Expected probabilities in [0,1] and positive floor required")
    return float((y-p).sum() / np.sqrt((p*(1-p)).sum() + variance_floor))


def require_features(features: Mapping[str, float], expected: list[str]) -> None:
    """Fail on naming mistakes; never silently fall back to another signal."""
    missing = set(expected) - set(features)
    if missing:
        raise KeyError(f"Feature registry mismatch: {sorted(missing)}")
    if not all(math.isfinite(float(features[k])) for k in expected):
        raise ValueError("Non-finite required feature")
