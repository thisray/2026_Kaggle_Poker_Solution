"""Window-safe, action-ordered opportunity features (small-panel reference).

Run large joins on GB10 with the provided DuckDB SQL. This module needs only
numpy/pandas and deliberately refuses to infer window membership from labels.
The normal p_enter policy must be fitted without the outer evaluation labels.
A window is an explicit list of hand indices, not an ID/file-order feature.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PAIR_KEY = ['window', 'pool', 'phase', 'p_lo', 'p_hi']

def aggression_before(aggressive: np.ndarray, offsets: np.ndarray, hand: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Count strictly earlier aggression; current action must never be subtracted."""
    g = np.asarray(aggressive, dtype=np.int64)
    off = np.asarray(offsets, dtype=np.int64)
    h, k = np.asarray(hand, dtype=np.int64), np.asarray(action, dtype=np.int64)
    if h.shape != k.shape or np.any(h < 0) or np.any(h >= len(off)-1):
        raise ValueError('Invalid hand/action index shapes')
    if np.any(k < off[h]) or np.any(k >= off[h+1]):
        raise ValueError('Action outside its hand')
    cumulative = np.r_[0, np.cumsum(g)]
    return cumulative[k] - cumulative[off[h]]

def align_to(source: pd.DataFrame, target: pd.DataFrame, values: np.ndarray, keys: list[str]) -> np.ndarray:
    if source.duplicated(keys).any() or target.duplicated(keys).any():
        raise ValueError('Duplicate alignment keys')
    if len(source) != len(values): raise ValueError('Source/value length mismatch')
    take = pd.MultiIndex.from_frame(source[keys]).get_indexer(pd.MultiIndex.from_frame(target[keys]))
    if np.any(take < 0): raise ValueError('Target keys absent from source')
    return np.asarray(values)[take]

def build_opportunities(decisions: pd.DataFrame, membership: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    required = {'h','player','pool','phase','act_idx','entered','p_enter'}
    if required - set(decisions): raise ValueError(f'Missing decisions: {required-set(decisions)}')
    if {'window','h'} - set(membership): raise ValueError('Membership requires window,h')
    if set(PAIR_KEY) - set(pairs): raise ValueError('Pairs require explicit window,pool,phase,endpoints')
    if decisions.duplicated(['h','player']).any(): raise ValueError('Multiple first decisions for a player-hand')
    if membership.duplicated(['window','h']).any(): raise ValueError('Duplicate window membership')
    if pairs.duplicated(PAIR_KEY).any(): raise ValueError('Duplicate pair-window')
    if np.any(pairs.p_lo >= pairs.p_hi): raise ValueError('Canonical unordered endpoints required')
    if not decisions.p_enter.between(0,1,inclusive='neither').all(): raise ValueError('p_enter must lie strictly in (0,1)')
    if not decisions.entered.isin([0,1]).all(): raise ValueError('entered must be binary')
    # This reference joins within each hand. The DuckDB equivalent is scalable.
    d = decisions.merge(membership[['window','h']],on='h',how='inner',validate='many_to_many')
    cols = ['window','h','pool','phase']
    q = d.merge(d,on=cols,suffixes=('_a','_b'))
    q = q[q.player_a < q.player_b].rename(columns={'player_a':'p_lo','player_b':'p_hi'})
    q = q.merge(pairs[PAIR_KEY],on=PAIR_KEY,how='inner',validate='many_to_one')
    if (q.act_idx_a == q.act_idx_b).any(): raise ValueError('Two actors share one action index')
    a_first = (q.act_idx_a < q.act_idx_b).to_numpy()
    out = q[PAIR_KEY+['h']].copy()
    for dest,ca,cb in [('first_player','p_lo','p_hi'),('second_player','p_hi','p_lo'),
                       ('first_action','act_idx_a','act_idx_b'),('second_action','act_idx_b','act_idx_a'),
                       ('y_first','entered_a','entered_b'),('y_second','entered_b','entered_a'),
                       ('p_first','p_enter_a','p_enter_b'),('p_second','p_enter_b','p_enter_a')]:
        out[dest] = np.where(a_first,q[ca],q[cb])
    # First entry is observed before the second decision: a predictable gate.
    out['opportunity'] = out.y_first.astype(float)
    out['observed'] = out.opportunity * out.y_second
    out['expected'] = out.opportunity * out.p_second
    out['residual'] = out.observed - out.expected
    out['variance_proxy'] = out.opportunity * out.p_second * (1-out.p_second)
    out['centered_cross'] = (out.y_first-out.p_first)*(out.y_second-out.p_second)
    # Kept as an explicitly NON-joint-null legacy diagnostic only.
    out['legacy_product_residual'] = out.y_first*out.y_second-out.p_first*out.p_second
    return out.reset_index(drop=True)

def aggregate(opportunities: pd.DataFrame, pairs: pd.DataFrame, shrink: float=25.) -> pd.DataFrame:
    if shrink <= 0: raise ValueError('shrink must be positive')
    if opportunities.duplicated(PAIR_KEY+['h']).any(): raise ValueError('Duplicate pair-hand exposure')
    a=opportunities.groupby(PAIR_KEY,sort=False).agg(
        observed_decisions=('h','size'),n_opportunities=('opportunity','sum'),
        observed=('observed','sum'),expected=('expected','sum'),residual=('residual','sum'),
        variance_proxy=('variance_proxy','sum'),centered_cross=('centered_cross','sum'),
        legacy_product_residual=('legacy_product_residual','sum')).reset_index()
    # No exposed decisions is recorded, never silently confused with clean behavior.
    out=pairs[PAIR_KEY].merge(a,on=PAIR_KEY,how='left',validate='one_to_one')
    out['no_decisions']=out.observed_decisions.isna().astype(np.int8)
    metrics=[c for c in a if c not in PAIR_KEY]
    out[metrics]=out[metrics].fillna(0.)
    out['shrunk_rate']=out.residual/(out.n_opportunities+shrink)
    out['z_proxy']=out.residual/np.sqrt(out.variance_proxy+1.)
    out['opportunity_fraction']=out.n_opportunities/out.observed_decisions.clip(lower=1)
    return out

def read_table(p: str) -> pd.DataFrame:
    return pd.read_parquet(p) if p.endswith('.parquet') else pd.read_csv(p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--decisions',required=True);p.add_argument('--membership',required=True);p.add_argument('--pairs',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    pairs=read_table(args.pairs);o=build_opportunities(read_table(args.decisions),read_table(args.membership),pairs)
    o.to_csv(out/'opportunities.csv.gz',index=False)
    aggregate(o,pairs).to_csv(out/'pair_features.csv.gz',index=False)
if __name__=='__main__':main()
