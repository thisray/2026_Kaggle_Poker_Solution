"""Choose a gameplay-risk compute budget and patch only complete evidence lists.
This does not submit, assign labels, or change risk/behavior. Low-risk pairs keep
all baseline evidence. Evaluate the identical gate and fallback out of fold.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

EVID = [f'evidence_hand_{i}' for i in range(1, 6)]


def select_slots(risk: pd.DataFrame, budget: int) -> pd.DataFrame:
    if budget < 1 or risk.slot.duplicated().any():
        raise ValueError('Positive budget and unique slot risk table required')
    if not np.isfinite(risk.risk_score.to_numpy(float)).all():
        raise ValueError('Nonfinite risk')
    # Do not select an arbitrary portion of a tied-risk bucket by identifiers.
    cutoff = risk.risk_score.nlargest(min(budget, len(risk))).min()
    cols = ['slot'] + (['pair_id'] if 'pair_id' in risk else []) + ['risk_score']
    return risk.loc[risk.risk_score >= cutoff, cols].copy()


def patch_frame(base: pd.DataFrame, scored: pd.DataFrame, membership: pd.DataFrame,
                pair_map: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    if pair_map is not None and 'pair_id' not in scored:
        if pair_map.slot.duplicated().any() or pair_map.pair_id.duplicated().any():
            raise ValueError('Pair mapping must be one-to-one')
        scored = scored.merge(pair_map[['slot', 'pair_id']], on='slot', how='left', validate='many_to_one')
    if scored.pair_id.isna().any() or base.pair_id.duplicated().any():
        raise ValueError('Incomplete pair IDs or duplicate baseline pairs')
    if not set(scored.pair_id) <= set(base.pair_id):
        raise ValueError('Patch includes unknown pairs')
    if scored.duplicated(['pair_id', 'hand_id']).any() or not np.isfinite(scored.score.astype(float)).all():
        raise ValueError('Duplicate candidates or nonfinite scores')
    if scored.groupby('pair_id').size().min() < 5:
        raise ValueError('Every patched pair requires five distinct candidates')
    # IDs serve as join keys and deterministic ties only; they are not features.
    top = scored.sort_values(['pair_id', 'score', 'hand_id'], ascending=[True, False, True], kind='stable').groupby('pair_id', sort=False).head(5)
    mem = membership[['pair_id', 'hand_id', 'phase']].drop_duplicates()
    if mem.duplicated(['pair_id', 'hand_id']).any():
        raise ValueError('Ambiguous phase in membership')
    check = top.merge(mem, on=['pair_id', 'hand_id'], how='left', validate='one_to_one')
    if not check.phase.eq('evaluation').all():
        raise ValueError('Patch evidence is not a shared evaluation hand')
    out = base.copy(); ix = out.set_index('pair_id').index
    positions = {p: i for i, p in enumerate(ix)}
    changed = 0
    for pair, g in top.groupby('pair_id', sort=False):
        row = positions[pair]; hands = g.hand_id.astype(str).tolist()
        changed += (out.iloc[row][EVID].astype(str).tolist() != hands)
        for c, hand in zip(EVID, hands):
            out.iat[row, out.columns.get_loc(c)] = hand
    if not out[['pair_id', 'risk_score', 'predicted_behavior']].equals(base[['pair_id', 'risk_score', 'predicted_behavior']]):
        raise AssertionError('Risk/behavior changed')
    return out, dict(patched_pairs=int(top.pair_id.nunique()), changed_pairs=int(changed),
                     fallback_pairs=int(len(base) - top.pair_id.nunique()), risk_behavior_preserved=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('select'); s.add_argument('--risk', required=True); s.add_argument('--budget', type=int, required=True); s.add_argument('--out', required=True)
    s = sub.add_parser('patch')
    for name in ['base', 'scored', 'membership', 'out']: s.add_argument('--' + name, required=True)
    s.add_argument('--pair-map')
    a = p.parse_args()
    if Path(a.out).exists(): raise FileExistsError(a.out)
    if a.command == 'select':
        select_slots(pd.read_csv(a.risk), a.budget).to_csv(a.out, index=False); return
    base = pd.read_csv(a.base, dtype=str); scored = pd.read_csv(a.scored, dtype={'pair_id': str, 'hand_id': str})
    mem = pd.read_csv(a.membership, dtype={'pair_id': str, 'hand_id': str})
    mapping = pd.read_csv(a.pair_map, dtype={'pair_id': str}) if a.pair_map else None
    out, receipt = patch_frame(base, scored, mem, mapping); out.to_csv(a.out, index=False)
    receipt.update(base_sha256=hashlib.sha256(Path(a.base).read_bytes()).hexdigest(),
                   output_sha256=hashlib.sha256(Path(a.out).read_bytes()).hexdigest(),
                   note='membership must be derived from raw seats and phase, not invented from predictions')
    Path(a.out + '.receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))

if __name__ == '__main__': main()
