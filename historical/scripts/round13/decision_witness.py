"""Role x street event witnesses over the supplied replay records.

This is a NEW representation, not a fitted detector. It computes nonlinear
same-decision interactions before pooling and keeps the two actor orientations.
Normal-policy probabilities must be cross-fitted upstream. Exact private-card
quantities are retrospective investigation features, NOT actor information.
Raising opportunity is the legacy proxy, not a certified poker rules engine.
"""
from __future__ import annotations
from typing import Iterable
import numpy as np

EVENTS = (
    'facing_partner', 'facing_outsider', 'heads_up_partner',
    'aggression_with_outsider', 'outsider_fold_after_both',
    'response_after_outsider_exit',
)
VALUES = (
    'price_pot', 'amount_pot', 'log_price_bb', 'log_amount_bb',
    'own_category', 'partner_category', 'category_gap',
    'hu_equity', 'hu_known', 'eq96', 'eq96_known',
    'policy_surprise', 'policy_known', 'fold_residual', 'call_residual',
    'aggression_residual', 'raise_opportunity_proxy',
    'fold_equity_price', 'call_disadvantage_amount',
    'check_value_opportunity', 'aggression_weakness_size',
    'terminal_call_minus_fold_ev', 'terminal_ev_valid',
    'fold_positive_terminal_ev', 'call_negative_terminal_ev',
    'partner_category_x_action_surprise', 'outsider_exit_count_same_street',
)
STATS = ('mean', 'max', 'min')
ROLE_NAMES = ('actor_A', 'actor_B', 'outsider')

def feature_names() -> list[str]:
    names = []
    for role in ROLE_NAMES:
        for street in range(4):
            for event in EVENTS:
                prefix = f'{role}/street{street}/{event}'
                names.append(prefix + '/log_count')
                names.extend(f'{prefix}/{stat}/{v}' for stat in STATS for v in VALUES)
    return names


def event_values(r: dict, partner: int | None, outsiders: list[int], exits: int) -> np.ndarray:
    """All products refer to this same decision. Unknown quantities have masks."""
    i = int(r['i']); bb = float(r['bb'])
    if bb <= 0: raise ValueError('Non-positive big blind')
    price, amount, pot = float(r['tc']), float(r['amt']), float(r['pot'])
    if min(price, amount, pot) < -1e-7: raise ValueError('Negative monetary field')
    odds = price / max(pot + price, 1e-8)
    own = float(r['geom'][i, 0])
    other = float(r['geom'][partner, 0]) if partner is not None else -1.
    qh = float(r['hu'][i, partner]) if partner is not None else -1.
    q_known = float(0. <= qh <= 1.)
    eq = float(r['eq96']); eq_known = float(0. <= eq <= 1.)
    prob_known = bool(r['prob_known'])
    probs = np.asarray(r['probs'], dtype=float)
    if prob_known and (len(probs) != 4 or not np.isfinite(probs).all() or
                       np.any(probs < 0) or abs(probs.sum() - 1) > 1e-4):
        raise ValueError('Invalid upstream action probabilities')
    action = int(r['y'])
    # Never turn a uniform missing-probability fallback into observed surprise.
    surprise = -np.log(max(float(probs[action]), 1e-6)) if prob_known else 0.
    residual = np.eye(4)[action] - probs if prob_known else np.zeros(4)
    pair = partner is not None
    toward = pair and int(r['la']) == partner and price > 0
    hu = pair and bool(r['alive'][partner]) and int(np.sum(r['alive'])) == 2
    post = np.asarray(r['total'], float).copy(); post[i] += price
    # Conservative equal-contestable-pot mask; excludes complex side-pot cases.
    simple = pair and abs(post[i] - post[partner]) < 1e-6 and all(post[o] <= post[i] + 1e-6 for o in outsiders)
    terminal = bool(simple and toward and hu and int(r['st']) == 3 and
                    float(r['stack']) >= price and q_known)
    ev = (qh * (pot + price) - price) / bb if terminal else 0.
    q = qh if q_known else 0.
    weak = 1. - q if q_known else 0.
    out = [odds, amount / max(bb, pot), np.log1p(price / bb), np.log1p(amount / bb),
           own, other, own - other if pair else 0., q, q_known,
           eq if eq_known else 0., eq_known, surprise, float(prob_known),
           residual[0], residual[2], residual[3], float(r['raise_proxy']),
           float(action == 0) * q * price / bb,
           float(action == 2) * max(odds - qh, 0.) * amount / bb if q_known else 0.,
           float(action == 1) * q * float(r['raise_proxy']),
           float(bool(r['aggr'])) * weak * amount / max(bb, pot),
           ev, float(terminal), float(action == 0) * max(ev, 0.),
           float(action == 2) * max(-ev, 0.),
           max(other, 0.) * surprise if pair else 0., float(exits)]
    result = np.asarray(out, np.float32)
    if result.size != len(VALUES) or not np.isfinite(result).all():
        raise ValueError('Invalid witness vector')
    return result


def oriented_witness(records: Iterable[dict], A: int, B: int) -> np.ndarray:
    if A == B or not 0 <= A < 6 or not 0 <= B < 6: raise ValueError('Invalid seats')
    others = [s for s in range(6) if s not in (A, B)]
    bins = [[[[] for _ in EVENTS] for _ in range(4)] for _ in range(3)]
    street = -1; exits: set[int] = set()
    for r in records:
        st = int(r['st']); i = int(r['i'])
        if st not in range(4): raise ValueError('Invalid street')
        if st != street:
            if st < street: raise ValueError('Non-monotonic streets')
            street = st; exits.clear()
        partner = B if i == A else A if i == B else None
        role = 0 if i == A else 1 if i == B else 2
        pair = partner is not None
        facing = float(r['tc']) > 0
        toward = pair and facing and int(r['la']) == partner
        bothmask = (1 << A) | (1 << B)
        exited = (not pair and int(r['act']) == 0 and int(r['la']) in (A, B)
                  and (int(r['faced'][i]) & bothmask) == bothmask)
        if exited: exits.add(i)
        masks = (
            toward,
            pair and facing and int(r['la']) in others,
            pair and bool(r['alive'][partner]) and int(np.sum(r['alive'])) == 2,
            pair and bool(r['aggr']) and bool(np.asarray(r['can'])[others].any()),
            exited,
            pair and toward and int(r['y']) in (0, 2) and len(exits) > 0,
        )
        values = event_values(r, partner, others, len(exits))
        for e, on in enumerate(masks):
            if on: bins[role][st][e].append(values)
    out = []
    for role_bins in bins:
        for street_bins in role_bins:
            for events in street_bins:
                out.append(np.log1p(len(events)))
                if events:
                    x = np.asarray(events)
                    out.extend(x.mean(0)); out.extend(x.max(0)); out.extend(x.min(0))
                else: out.extend(np.zeros(len(STATS) * len(VALUES)))
    return np.asarray(out, np.float32)


def two_views(records: Iterable[dict], A: int, B: int) -> np.ndarray:
    records = list(records)
    # The learner processes these views separately; average model predictions.
    # DO NOT average these feature vectors before a nonlinear learner.
    return np.stack([oriented_witness(records, A, B), oriented_witness(records, B, A)])
