"""R5 d4: what is the EVAL-side known-family E of the deployed r10_ci pipeline?
dev pairs see ~122 shared hands, eval pairs ~89 (2000 vs 3000 hands per phase),
and AP@5 depends strongly on exposure. Re-weighting the dev per-pair AP by the
eval exposure distribution pins E_known(eval), which in turn pins the fourth
family's share w and its AP through  E_pub = (1-w) E_known + w AP_F4."""
import numpy as np, pandas as pd, sys, os, json
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18')
import ci_censored_event as C
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'
full=C.prepare(pd.read_parquet(f'{O}/t5_dev_seq.parquet'))
cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
nhand=full.groupby('slot').size().rename('n')
# eval exposure of the pairs that actually carry the score: top-ranked eval pairs
z=C.prepare(pd.read_parquet(f'{O}/r4/z7_groups_eval_full.parquet'))
nev=z.groupby('slot').size()
print('dev n: mean %.1f median %.0f  |  eval(frame) n: mean %.1f median %.0f  n_pairs %d'%(nhand.mean(),nhand.median(),nev.mean(),nev.median(),len(nev)))
rows=[]
for fam,s in full.groupby('fam'):
    s=s.reset_index(drop=True); counts=s.groupby('slot').ev.sum()
    c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev']],on=['slot','h'],validate='one_to_one').reset_index(drop=True)
    a=C.pair_ap(c,-c.r.values,counts)   # R15 evidence, the deployed DT/SP choice in r10_ci
    n=nhand.reindex(a.index)
    # weight dev pairs so that their n distribution matches the eval frame
    bins=np.array([0,70,85,100,115,130,150,1e9])
    dw=pd.cut(n,bins).value_counts(normalize=True).sort_index()
    ew=pd.cut(nev,bins).value_counts(normalize=True).sort_index()
    w=(ew/dw.replace(0,np.nan)).reindex(pd.cut(n,bins)).values; w=np.where(np.isfinite(w),w,0.0)
    rows.append(dict(fam=fam,pairs=len(a),dev_R15=float(a.mean()),
        evalw_R15=float(np.average(a.values,weights=w)) if w.sum()>0 else np.nan,
        q1=float(a[n<=np.percentile(n,33)].mean()),q2=float(a[(n>np.percentile(n,33))&(n<=np.percentile(n,67))].mean()),q3=float(a[n>np.percentile(n,67)].mean()),
        n_med=float(n.median())))
d=pd.DataFrame(rows); print(d.to_string())
fr={'directed_transfer':148/372,'soft_play':132/372,'coordinated_isolation':92/372}
for col in ('dev_R15','evalw_R15'):
    print(col,'weighted known-family E =',round(float(sum(fr[r.fam]*r[col] for _,r in d.iterrows())),4))
