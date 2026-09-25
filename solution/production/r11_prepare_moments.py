"""Build the historical Round11 two-view moments from raw replay arrays."""
from __future__ import annotations
import argparse, hashlib, importlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd


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


def tab_view(records, A: int, B: int, n_moments: int = 24) -> np.ndarray:
    """Equivalent to Round8 replay.view()[3], without num/cat/cards allocation."""
    outsiders = [s for s in range(6) if s not in (A, B)]
    events = [[] for _ in range(6)]
    exit_same = 0
    for r in records:
        i = r['i']; j = B if i == A else A if i == B else -1
        pair = j >= 0; bb = r['bb']
        toward = pair and r['la'] == j and r['tc'] > 0
        bothmask = (1 << A) | (1 << B)
        same = sum((int(r['faced'][o]) & bothmask) == bothmask for o in outsiders)
        ncouts = int(r['can'][outsiders].sum())
        fold_both = (not pair and r['act'] == 0 and r['la'] in (A, B)
                     and (int(r['faced'][i]) & bothmask) == bothmask)
        if fold_both:
            exit_same += 1
        pj = bool(r['alive'][j]) if pair else False
        q = float(r['full'][i]); qh = float(r['hu'][i, j]) if pair else -1.
        g = r['geom'][i]; pg = r['geom'][j] if pair else np.full(10, -1.)
        post = r['total'].copy(); post[i] += r['tc']
        simple = (pair and abs(post[i] - post[j]) < 1e-6
                  and all(post[o] <= post[i] + 1e-6 for o in outsiders))
        terminal = (simple and toward and r['st'] == 3 and r['alive'].sum() == 2
                    and r['stack'] >= r['tc'] and qh >= 0)
        ev = (qh * (r['pot'] + r['tc']) - r['tc']) / bb if terminal else 0.
        moment = [q, qh, r['eq96'], q - r['eq96'] if q >= 0 and r['eq96'] >= 0 else 0.,
                  ev, r['tc'] / max(1e-8, r['pot'] + r['tc']),
                  r['amt'] / max(bb, r['pot']), r['raise_proxy'],
                  r['probs'][3] * r['raise_proxy'], float(same), *g,
                  pg[0], pg[6], g[0] - pg[0], max(qh, 0) * r['tc'] / bb]
        if len(moment) != n_moments:
            raise ValueError('Legacy moment schema has changed')
        mask = [pair and toward and r['act'] == 0,
                pair and toward and r['y'] == 2,
                pair and r['act'] == 1 and pj and r['alive'].sum() == 2,
                pair and r['aggr'] and ncouts > 0, fold_both,
                pair and toward and r['y'] in (0, 2) and exit_same > 0]
        for k, selected in enumerate(mask):
            if selected:
                events[k].append(moment)
    result = []
    for event in events:
        x = np.asarray(event, float) if event else np.zeros((1, n_moments))
        result.extend(x.mean(0)); result.extend(x.max(0))
    result.extend(len(event) for event in events)
    return np.asarray(result, np.float32)


def symmetry_moments(records, A, B, n_moments=24):
    t = np.stack([tab_view(records, A, B, n_moments), tab_view(records, B, A, n_moments)])
    x = np.r_[t.mean(0), t.max(0)]
    return np.nan_to_num(x, nan=0., posinf=30., neginf=-30.).clip(-30, 30).astype(np.float32)


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
    n_mom = len(legacy.MOMENT_NAMES); dim = 2 * len(legacy.TAB_NAMES)
    x = np.lib.format.open_memmap(out / 'moments.npy', mode='w+', dtype='float32', shape=(len(d), dim))
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
            x[row] = symmetry_moments(records, int(aa[0]), int(bb[0]), n_mom)
        if count % 1000 == 0:
            print(f'moments {count}/{len(by_hand)} unique hands', flush=True)
    x.flush(); d.to_csv(out / 'meta.csv', index=False)
    receipt = dict(rows=len(d), pairs=len(slots), unique_hands=len(by_hand), dim=dim,
                   exact=exact, seconds=time.monotonic() - start, replay_checks=checks,
                   neural_tensors_created=False,
                   source_sha256=hashlib.sha256(Path(candidates).read_bytes()).hexdigest(),
                   legacy_replay_sha256=hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest(),
                   scope='same Round8 replay semantics; full real-data parity remains worker confirmation')
    (out / 'pack.json').write_text(json.dumps(receipt, indent=2)); (out / '_SUCCESS').touch()
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['candidates', 'np-dir', 'legacy-code', 'out']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--selected-slots'); p.add_argument('--pair-offset', type=int, default=0)
    p.add_argument('--pair-limit', type=int, default=0); p.add_argument('--exact', action='store_true')
    print(json.dumps(prepare(**vars(p.parse_args())), indent=2))
