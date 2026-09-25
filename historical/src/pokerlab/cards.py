"""Small exact hand evaluator and explicit-information equity reference.

Correctness-oriented Python implementation; not intended for 18M-action scans.
Higher rank tuples are BETTER. This differs from several third-party evaluators.
"""
from __future__ import annotations
from collections import Counter
from itertools import combinations
import math
import random
import re
from typing import Iterable

RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = tuple(r + s for r in RANKS for s in SUITS)
SUIT_MAP = str.maketrans({"♣": "c", "♦": "d", "♥": "h", "♠": "s"})


def parse_cards(value: str | Iterable[str]) -> tuple[str, ...]:
    raw = value if isinstance(value, str) else " ".join(value)
    raw = raw.translate(SUIT_MAP).replace("10", "T")
    raw = re.sub(r"[\s,;\[\]\(\)'\"]+", "", raw)
    if len(raw) % 2:
        raise ValueError("Invalid card text length")
    cards = tuple(raw[i].upper() + raw[i+1].lower() for i in range(0, len(raw), 2))
    if any(c not in DECK for c in cards) or len(set(cards)) != len(cards):
        raise ValueError("Invalid or duplicate cards")
    return cards


def board_at_street(board: str | Iterable[str], street: str) -> tuple[str, ...]:
    cards = parse_cards(board)
    normalized = str(street).lower().replace("_", "").replace("-", "")
    sizes = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
    if normalized not in sizes or len(cards) > 5:
        raise ValueError("Unknown street or board longer than five")
    n = sizes[normalized]
    if len(cards) < n:
        raise ValueError("Board is shorter than the requested street")
    return cards[:n]


def rank5(cards: str | Iterable[str]) -> tuple[int, ...]:
    c = parse_cards(cards)
    if len(c) != 5:
        raise ValueError("rank5 needs exactly five cards")
    vals = [RANKS.index(x[0]) + 2 for x in c]
    counts = Counter(vals)
    ordered = sorted(((n, v) for v, n in counts.items()), reverse=True)
    unique = sorted(counts, reverse=True)
    straight_high = 0
    if len(unique) == 5:
        if unique[0] - unique[-1] == 4:
            straight_high = unique[0]
        elif unique == [14, 5, 4, 3, 2]:
            straight_high = 5
    flush = len({x[1] for x in c}) == 1
    if flush and straight_high:
        return (8, straight_high)
    if ordered[0][0] == 4:
        return (7, ordered[0][1], ordered[1][1])
    if [x[0] for x in ordered] == [3, 2]:
        return (6, ordered[0][1], ordered[1][1])
    if flush:
        return (5, *sorted(vals, reverse=True))
    if straight_high:
        return (4, straight_high)
    if ordered[0][0] == 3:
        return (3, ordered[0][1], *sorted((v for v in vals if counts[v] == 1), reverse=True))
    if [x[0] for x in ordered[:2]] == [2, 2]:
        return (2, *sorted((v for v, n in counts.items() if n == 2), reverse=True),
                next(v for v, n in counts.items() if n == 1))
    if ordered[0][0] == 2:
        return (1, ordered[0][1], *sorted((v for v, n in counts.items() if n == 1), reverse=True))
    return (0, *sorted(vals, reverse=True))


def best_rank(cards: str | Iterable[str]) -> tuple[int, ...]:
    c = parse_cards(cards)
    if not 5 <= len(c) <= 7:
        raise ValueError("best_rank supports five through seven cards")
    return max(rank5(x) for x in combinations(c, 5))


def heads_up_equity(hero: str | Iterable[str], villain: str | Iterable[str],
                    board: str | Iterable[str], samples: int = 2000,
                    seed: int = 2026, exact_limit: int = 5000) -> dict[str, float | int | bool]:
    """Equity with BOTH players' actual cards known; uniform legal runouts.

    This is retrospective conditional equity, NOT a player-information-set model,
    not a range model, and not multiway/side-pot equity. Exact if combinations
    fit exact_limit; otherwise Monte Carlo sampling with replacement over runouts.
    """
    h, v, b = parse_cards(hero), parse_cards(villain), parse_cards(board)
    if len(h) != 2 or len(v) != 2 or len(b) > 5:
        raise ValueError("Need two hole cards per player and at most five board cards")
    known = (*h, *v, *b)
    if len(set(known)) != len(known):
        raise ValueError("Cards overlap between players or board")
    if samples < 1 or exact_limit < 1:
        raise ValueError("Positive samples and exact_limit required")
    remaining = [c for c in DECK if c not in known]
    missing = 5 - len(b)
    space = math.comb(len(remaining), missing)
    exact = space <= exact_limit
    rng = random.Random(seed)
    runouts = combinations(remaining, missing) if exact else (
        tuple(rng.sample(remaining, missing)) for _ in range(samples))
    values = []
    for extra in runouts:
        a, z = best_rank((*h, *b, *extra)), best_rank((*v, *b, *extra))
        values.append(1.0 if a > z else 0.5 if a == z else 0.0)
    mean = sum(values) / len(values)
    se = 0.0 if exact or len(values) == 1 else math.sqrt(
        sum((x - mean)**2 for x in values) / (len(values) - 1) / len(values))
    return {"equity": mean, "standard_error": se, "evaluated_runouts": len(values), "exact": exact}


def heads_up_call_ev(equity: float, pot_before: float, call_cost: float) -> float:
    """Incremental call EV vs folding; HU, no future bets, no side pot/rake."""
    if not all(math.isfinite(v) for v in (equity, pot_before, call_cost)):
        raise ValueError("Inputs must be finite")
    if not 0 <= equity <= 1 or pot_before < 0 or call_cost < 0:
        raise ValueError("Invalid equity, pot, or call cost")
    return equity * (pot_before + call_cost) - call_cost
