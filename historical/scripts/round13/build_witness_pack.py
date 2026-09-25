"""Build NEW candidate witness features on GB10 using existing exported arrays.
No training, submission, network writes, or overwrite. The real-data builder has
not run in the review environment; unit tests cover the extractor only.
"""
from __future__ import annotations
import argparse, hashlib, importlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from decision_witness import two_views, feature_names

def load_legacy(code_dir: str):
    path = str(Path(code_dir).resolve())
    sys.path.insert(0, path)
    replay = importlib.import_module('replay')
    if Path(replay.__file__).resolve().parent != Path(path):
        raise RuntimeError('Another replay module is already loaded; use a fresh process')
    return replay


def read_table(path):
    p = Path(path)
    if not p.exists() and p.with_suffix('.csv').exists():
        p = p.with_suffix('.csv')
    return pd.read_parquet(p) if p.suffix == '.parquet' else pd.read_csv(p)


def prepare(candidates, np_dir, legacy_code, out, selected_slots=None, pair_offset=0,
            pair_limit=0, exact=False):
    start = time.monotonic(); out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Use a new output directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    d = read_table(candidates)
    required = {'slot', 'pool', 'pair_player_lo', 'pair_player_hi', 'hand_id', 'u_r5b'}
    if required - set(d):
        raise ValueError(f'Missing columns: {sorted(required - set(d))}')
    if d.duplicated(['slot', 'hand_id']).any():
        raise ValueError('Duplicate candidate keys')
    if selected_slots is not None:
        wanted = set(read_table(selected_slots)['slot'])
        absent = wanted - set(d.slot)
        if absent:
            raise ValueError(f'{len(absent)} requested slots have no candidates')
        d = d[d.slot.isin(wanted)]
    slots = np.sort(d.slot.unique())
    slots = slots[pair_offset:(pair_offset + pair_limit) if pair_limit else None]
    d = d[d.slot.isin(slots)].sort_values(['slot', 'hand_id'], kind='stable').reset_index(drop=True)
    if d.empty:
        raise ValueError('Empty candidate set')
    if 'fold' in d and d.groupby('pool').fold.nunique().max() > 1:
        raise ValueError('Pool spans folds')
    legacy = load_legacy(legacy_code); a = legacy.Arrays(np_dir)
    hi = read_table(Path(np_dir) / 'hand_index.parquet').set_index('hand_id').hi
    pi = read_table(Path(np_dir) / 'player_index.parquet').set_index('player_id').pi
    h = d.hand_id.map(hi); pa = d.pair_player_lo.map(pi); pb = d.pair_player_hi.map(pi)
    if h.isna().any() or pa.isna().any() or pb.isna().any():
        raise ValueError('ID mapping incomplete')
    h = h.to_numpy(int); pa = pa.to_numpy(int); pb = pb.to_numpy(int)
    if 'ev' in d and not np.all(a['h_phase'][h] == 0):
        raise ValueError('Development labels on evaluation hands')
    dim = len(feature_names())
    x = np.lib.format.open_memmap(out / 'witness_views.npy', mode='w+', dtype='float32', shape=(len(d), 2, dim))
    by_hand = {}
    for row, hand in enumerate(h):
        by_hand.setdefault(int(hand), []).append(row)
    checks = {'max_stack_error': 0., 'illegal_actors': 0}
    for count, (hand, rows) in enumerate(by_hand.items(), 1):
        records, _, _, stats = legacy.replay_hand(hand, a, exact=exact)
        checks['max_stack_error'] = max(checks['max_stack_error'], stats['max_stack_error'])
        checks['illegal_actors'] += stats['illegal_actors']
        for row in rows:
            aa = np.flatnonzero(a['s_player'][hand] == pa[row]); bb = np.flatnonzero(a['s_player'][hand] == pb[row])
            if len(aa) != 1 or len(bb) != 1 or aa[0] == bb[0]:
                raise ValueError('Pair is not shared in hand')
            x[row] = two_views(records, int(aa[0]), int(bb[0]))
        if count % 1000 == 0:
            print(f'witness {count}/{len(by_hand)} unique hands', flush=True)
    x.flush(); (out / 'feature_names.json').write_text(json.dumps(feature_names(), indent=2)); d.to_csv(out / 'meta.csv', index=False)
    receipt = dict(rows=len(d), pairs=len(slots), unique_hands=len(by_hand), dim=dim,
                   exact=exact, seconds=time.monotonic() - start, replay_checks=checks,
                   neural_tensors_created=False,
                   source_sha256=hashlib.sha256(Path(candidates).read_bytes()).hexdigest(),
                   legacy_replay_sha256=hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest(),
                   scope='NEW role/street witnesses; synthetic tests only here; requires real-data validation on GB10',
                   orientation_policy='train both views and average predictions, never pool raw orientations',
                   prob_known=a['probs'] is not None,
                   extractor_sha256=hashlib.sha256(Path(__file__).with_name('decision_witness.py').read_bytes()).hexdigest())
    (out / 'pack.json').write_text(json.dumps(receipt, indent=2)); (out / '_SUCCESS').touch()
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['candidates', 'np-dir', 'legacy-code', 'out']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--selected-slots'); p.add_argument('--pair-offset', type=int, default=0)
    p.add_argument('--pair-limit', type=int, default=0); p.add_argument('--exact', action='store_true')
    print(json.dumps(prepare(**vars(p.parse_args())), indent=2))
