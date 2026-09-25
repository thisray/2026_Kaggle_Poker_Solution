"""Verified action semantics used by the real-data feature adapter."""
from __future__ import annotations

import math
from typing import Literal

from .cards import board_at_street


AllInKind = Literal["not_all_in", "all_in_call", "all_in_aggressive"]


def normalize_action(action: str) -> str:
    """Normalize source spellings without changing the action meaning."""
    value = str(action).strip().lower().replace("-", "_").replace(" ", "_")
    if not value:
        raise ValueError("Action cannot be blank")
    return value


def verified_increment(amount: float, amount_to: float | None = None, tolerance: float = 1e-9) -> float:
    """Return an incremental amount after checking an optional street total.

    ``amount_to`` is a cumulative street target when present. It is checked for
    consistency but is never silently added to the increment or treated as a
    hand-level ledger total.
    """
    if not math.isfinite(float(amount)) or float(amount) < 0:
        raise ValueError("Incremental amount must be finite and non-negative")
    if amount_to is not None:
        if not math.isfinite(float(amount_to)) or float(amount_to) < 0:
            raise ValueError("Cumulative amount must be finite and non-negative")
        if float(amount_to) + tolerance < float(amount):
            raise ValueError("Cumulative amount cannot be below the increment")
    return float(amount)


def classify_all_in(action: str, amount: float, to_call: float, tolerance: float = 1e-9) -> AllInKind:
    """Classify an all-in as a call or price-increasing action.

    The classification answers price change only. It does not claim that a
    short all-in reopens betting or that the action is strategically aggressive.
    """
    normalized = normalize_action(action)
    increment = verified_increment(amount)
    if not math.isfinite(float(to_call)) or float(to_call) < 0:
        raise ValueError("Call cost must be finite and non-negative")
    if normalized not in {"all_in", "allin"}:
        return "not_all_in"
    return "all_in_call" if increment <= float(to_call) + tolerance else "all_in_aggressive"


def decision_board(board_cards: str, street: str) -> tuple[str, ...]:
    """Return only the public board cards available at the decision street."""
    return board_at_street(board_cards, street)


__all__ = ["AllInKind", "classify_all_in", "decision_board", "normalize_action", "verified_increment"]
