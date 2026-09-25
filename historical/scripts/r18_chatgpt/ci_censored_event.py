#!/usr/bin/env python3
"""R18 CI-only full-hand event model, with right-censored training labels.

No submission/network side effects. IDs are keys, never predictive features.
Input full hands: R3 t5_dev_seq schema (sl/slot, h, ts, fam, ev, gameplay).
Input candidates: R3 t4_wrong_vs_hit schema (slot,h,r); r=1 is best R15.
Run `evaluate` for pool OOF; `fit` for a final model; `infer` writes a small
pair/evidence patch from ALL shared hands plus the existing candidate set.
See ../docs/05_重現與GB10接軌.md for GB10 extraction and integration.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

FAMILY = 'coordinated_isolation'
FEATURES = ['actor_net','partner_net','actor_win','partner_win','actor_fold','partner_fold',
 'pair_net','sd_any','iso','pair_win','xfer','own','par','y1','y2','o_fold_after',
 'o_in_at_trig','o_callraise_after','end_pre','o_alive_endpre','o_vol_before','pa_at_trig',
 'out_won','out_sd','all_out_folded','both_flop','card_gap','all_in_at_trigger','o_fold_fraction']
PARAMS = dict(n_estimators=400,learning_rate=.035,num_leaves=15,min_child_samples=40,
 reg_lambda=20,colsample_bytree=.85,verbosity=-1,n_jobs=2,random_state=27)

def read_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == '.csv':
        return pd.read_csv(path)
    return pd.read_parquet(path)

def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    s = frame.rename(columns={'sl':'slot'}).copy().reset_index(drop=True)
    if s.duplicated(['slot','h']).any():
        raise ValueError('Expected one row per (slot,h).')
    s['pool'] = s.slot // 900  # opaque pool grouping, excluded from model inputs
    for stem,a,b in [('net','na','nb'),('win','wa','wb'),('fold','fa','fb')]:
        s['actor_'+stem] = np.where(s.who==0,s[a],s[b])
        s['partner_'+stem] = np.where(s.who==0,s[b],s[a])
    s['card_gap'] = s.par-s.own
    # This means six active players, NOT a poker all-in. Name retained for R18 reproduction.
    s['all_in_at_trigger'] = s.pa_at_trig==6
    s['o_fold_fraction'] = s.o_fold_after/(1+s.o_in_at_trig)
    missing = set(FEATURES+['ts'])-set(s)
    if missing:
        raise ValueError(f'Missing gameplay columns: {sorted(missing)}')
    if not np.isfinite(s[FEATURES+['ts']].astype(float).to_numpy()).all():
        raise ValueError('Non-finite gameplay inputs; reproduce R3 missing-value conventions.')
    return s

def uncensored_training_rows(s: pd.DataFrame) -> np.ndarray:
    # A capped first-five evidence list does not label later hands as negatives.
    counts = s.groupby('slot').ev.sum()
    if (counts>5).any() or (counts<1).any():
        raise ValueError('This learner expects positive pairs with 1..5 listed hands.')
    last = s[s.ev.astype(bool)].groupby('slot').ts.max()
    return ((s.ts<=s.slot.map(last)) | (s.slot.map(counts)<5)).to_numpy()

def first_k_marginal(s: pd.DataFrame, event_probability: np.ndarray, k: int=5) -> np.ndarray:
    """P(hand is among the first k events), under conditional independence.
    The probabilities must cover every shared hand, not just the candidate top20.
    Sort by documented gameplay timestamp; hand index is only an exact-tie key.
    """
    p = np.asarray(event_probability, dtype=float)
    if len(p)!=len(s) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid event probabilities.')
    if not s.index.equals(pd.RangeIndex(len(s))):
        raise ValueError('Reset the full-hand frame index before decoding.')
    q = np.zeros(len(s),float)
    for _,g in s.sort_values(['slot','ts','h'],kind='stable').groupby('slot',sort=False):
        dist=np.zeros(k); dist[0]=1.
        for i in g.index:
            v=p[i]; q[i]=v*dist.sum()
            dist=np.r_[dist[0]*(1-v), dist[1:]*(1-v)+dist[:-1]*v]
    return q

def rank_candidates(full: pd.DataFrame, candidates: pd.DataFrame, q: np.ndarray,
                    weight: float=.5) -> pd.DataFrame:
    if not 0<=weight<=1:
        raise ValueError('weight must be in [0,1].')
    c=candidates.drop(columns=['ts','q','newscore'],errors='ignore').copy()
    if c.duplicated(['slot','h']).any():
        raise ValueError('Duplicate candidates.')
    j=c.merge(full[['slot','h','ts']].assign(q=q),on=['slot','h'],validate='one_to_one',how='left')
    if j[['ts','q']].isna().any().any():
        raise ValueError('Candidate missing from full shared-hand coverage.')
    qr=j.groupby('slot').q.rank(pct=True)
    br=(-j.r).groupby(j.slot).rank(pct=True)
    j['newscore']=(1-weight)*br+weight*qr
    return j

def pair_ap(candidates: pd.DataFrame, score: np.ndarray | pd.Series,
            counts: pd.Series) -> pd.Series:
    z=candidates[['slot','h','ts','ev']].copy(); z['score']=np.asarray(score)
    z=z.sort_values(['slot','score','ts','h'],ascending=[True,False,True,True],kind='stable')
    z['rank']=z.groupby('slot').cumcount()+1; z=z[z['rank']<=5]
    z['value']=z.ev.astype(float)*z.groupby('slot').ev.cumsum()/z['rank']
    return z.groupby('slot').value.sum().reindex(counts.index,fill_value=0)/counts.clip(upper=5)

def evaluate(full: pd.DataFrame, candidates: pd.DataFrame, seed: int=260919,
             threads: int=2) -> tuple[dict,pd.DataFrame,pd.DataFrame]:
    all_s=prepare(full)
    pools=np.array(sorted(all_s.pool.unique()))  # same all-positive-pool universe as the experiment
    s=all_s[all_s.fam==FAMILY].reset_index(drop=True)
    c=candidates.rename(columns={'sl':'slot'}).copy()
    c=c[c.slot.isin(s.slot)].reset_index(drop=True)
    # Source of all truth is the full-hand frame, not target coverage in top20.
    c=c.drop(columns=['ev','ts','y1','pa_at_trig'],errors='ignore').merge(s[['slot','h','ts','ev','y1','pa_at_trig']],on=['slot','h'],validate='one_to_one')
    if c.ev.isna().any():
        raise ValueError('Truth join failed.')
    include=uncensored_training_rows(s); p=np.zeros(len(s)); fmap={}
    folds=GroupKFold(5,shuffle=True,random_state=seed).split(pools,groups=pools)
    for fold,(_,va_pool) in enumerate(folds):
        vp=pools[va_pool]; tr=(~s.pool.isin(vp)).to_numpy()&include; va=s.pool.isin(vp).to_numpy()
        model=lgb.LGBMClassifier(**{**PARAMS,'n_jobs':threads})
        model.fit(s.loc[tr,FEATURES].astype(float),s.loc[tr,'ev'])
        p[va]=model.predict_proba(s.loc[va,FEATURES].astype(float))[:,1]
        fmap.update({int(pool):fold for pool in vp})
    q=first_k_marginal(s,p); j=rank_candidates(s,c,q)
    counts=s.groupby('slot').ev.sum(); base=pair_ap(j,-j.r,counts)
    hard=pair_ap(j,-j.r-100*(~((j.pa_at_trig==6)&j.y1.isin([2,3]))),counts) if {'pa_at_trig','y1'}<=set(j) else None
    # R3 rule requires six active players and first member action call/raise. Verify with exported ci_viol if present.
    if 'ci_viol' in j:
        hard=pair_ap(j,-j.r-100*j.ci_viol,counts)
    new=pair_ap(j,j.newscore,counts)
    per=pd.DataFrame({'R15':base,'R18':new,'pool':counts.index//900})
    if hard is not None: per['R3_hard']=hard
    reference=per['R3_hard'] if hard is not None else per.R15
    delta=per.R18-reference
    summary={'seed':seed,'n_pairs':len(counts),'n_full_hands':len(s),'n_evidence':int(counts.sum()),
      'R15_CI_E':float(base.mean()),'R18_CI_E':float(new.mean()),
      'delta_vs_R15':float((new-base).mean()),'reference':'R3_hard' if hard is not None else 'R15',
      'delta_vs_reference':float(delta.mean()),'better_pairs':int((delta>1e-12).sum()),
      'worse_pairs':int((delta< -1e-12).sum()),
      'fold_delta':delta.groupby(pd.Series([fmap[p] for p in counts.index//900],index=counts.index)).mean().tolist(),
      'validation':'Exploratory grouped OOF with frozen R15; repeated development use, not independent final holdout or LB.'}
    return summary,per,j

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['evaluate','fit','infer'])
    ap.add_argument('--full',required=True); ap.add_argument('--candidates'); ap.add_argument('--model')
    ap.add_argument('--out',required=True); ap.add_argument('--seed',type=int,default=260919); ap.add_argument('--threads',type=int,default=2)
    args=ap.parse_args(); out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    raw=read_frame(args.full)
    if args.mode=='evaluate':
        if not args.candidates: ap.error('--candidates is required')
        summary,per,rows=evaluate(raw,read_frame(args.candidates),args.seed,args.threads)
        out.write_text(json.dumps(summary,indent=2),encoding='utf-8')
        per.to_csv(out.with_suffix('.pairs.csv')); rows.to_csv(out.with_suffix('.candidates.csv'),index=False)
        print(json.dumps(summary,indent=2)); return
    s=prepare(raw)
    if args.mode=='fit':
        s=s[s.fam==FAMILY].reset_index(drop=True); tr=uncensored_training_rows(s)
        model=lgb.LGBMClassifier(**{**PARAMS,'n_jobs':args.threads})
        model.fit(s.loc[tr,FEATURES].astype(float),s.loc[tr,'ev'])
        model.booster_.save_model(str(out)); return
    if not args.model or not args.candidates: ap.error('infer needs --model and --candidates')
    if 'fam' in s and not s.fam.eq(FAMILY).all(): ap.error('infer must contain only CI-routed pairs')
    model=lgb.Booster(model_file=args.model)
    p=model.predict(s[FEATURES].astype(float),num_threads=args.threads)
    j=rank_candidates(s,read_frame(args.candidates),first_k_marginal(s,p))
    if 'pair_id' not in j or 'hand_id' not in j: ap.error('inference candidates need pair_id and hand_id keys')
    top=j.sort_values(['slot','newscore','ts','h'],ascending=[True,False,True,True],kind='stable').groupby('slot',sort=False).head(5)
    if not top.groupby('pair_id').size().eq(5).all(): raise ValueError('Fewer than five candidates.')
    top['rank']=top.groupby('pair_id').cumcount()+1
    patch=top.pivot(index='pair_id',columns='rank',values='hand_id')
    patch.columns=[f'evidence_hand_{i}' for i in patch.columns]; patch.reset_index().to_csv(out,index=False)
    print(f'Wrote CI-only patch: {len(patch)} pairs. No Kaggle upload performed.')

if __name__=='__main__': main()
