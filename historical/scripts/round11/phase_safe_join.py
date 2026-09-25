"""Prevent full-development/evaluation/subwindow feature conflation.
Subwindow features must be computed on exactly the same selected hands as the
base ptab. This join refuses to silently substitute full-dev features.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def attach_window_features(base: pd.DataFrame, feature_tables: dict[str,pd.DataFrame],
                           window_id: str) -> pd.DataFrame:
    keys=['pool','p_lo','p_hi']
    if window_id not in feature_tables:
        raise KeyError(f'No exact features for {window_id}; do not fall back to full dev')
    f=feature_tables[window_id]
    if f.duplicated(keys).any():raise ValueError('Duplicate feature keys within a window')
    cols=[c for c in f if c.startswith('b_')]
    if set(cols)&set(base):raise ValueError('Existing features would be overwritten')
    out=base.merge(f[keys+cols],on=keys,how='left',validate='many_to_one',indicator='_source_match',sort=False)
    if not out._source_match.eq('both').all():raise ValueError('Missing features; do not replace with zero silently')
    return out.drop(columns='_source_match')


def matched_auc(narrow: pd.DataFrame, exact: pd.DataFrame, mc_columns,
                keys=('pair_id','hand_id')) -> dict:
    if narrow.duplicated(list(keys)).any() or exact.duplicated(list(keys)).any():raise ValueError('Duplicate event keys')
    d=narrow.merge(exact[list(keys)+['eq_exact']],on=list(keys),how='inner',validate='one_to_one')
    cols=['eq_exact']+list(mc_columns)
    mask=np.isfinite(d[cols].to_numpy(float)).all(1)
    d=d.loc[mask]
    if len(d)==0 or d.ev.nunique()!=2:raise ValueError('No matched two-class cohort')
    return {'matched_rows':len(d),'AUC':{c:float(roc_auc_score(d.ev,d[c])) for c in cols},
            'scope':'same finite event cohort; still marginal diagnostic, not incremental MAP@5'}
