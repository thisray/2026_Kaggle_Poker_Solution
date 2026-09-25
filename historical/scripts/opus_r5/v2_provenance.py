"""R5-V2: full provenance of a final candidate against the LB-verified r13 base, and
against every patch file it is supposed to contain. Confirms that (a) the label column
is untouched, (b) every changed evidence row lies inside the pair set of exactly one
patch, (c) those rows are cell-exact with that patch, (d) nothing else moved.
usage: python v2_provenance.py <candidate.csv>"""
import pandas as pd, numpy as np, sys, hashlib, os
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'
EVC=[f'evidence_hand_{i}' for i in range(1,6)]
cf=sys.argv[1]; cur=pd.read_csv(cf,dtype=str,keep_default_na=False).set_index('pair_id')
base=pd.read_csv(f'{O}/r2_candidates/r13_ndwrank_cinew_f4.csv',dtype=str,keep_default_na=False).set_index('pair_id')
print(f'candidate {os.path.basename(cf)}  sha {hashlib.sha256(open(cf,"rb").read()).hexdigest()[:16]}')
assert (cur.index==base.index).all(), 'pair order differs'
print(f'  predicted_behavior identical to the LB-verified r13 base : {bool((cur.predicted_behavior.values==base.predicted_behavior.values).all())}')
print(f'  risk_score identical to r13                              : {bool((cur.risk_score.values==base.risk_score.values).all())}  (expected False: the fused ranking)')
rk=lambda d: d.risk_score.astype(float).rank(ascending=False,method='first')
ra,rb=rk(cur),rk(base)
for K in (250,450,600,1000):
    print(f'    top-{K} overlap with the LB-verified NDw ranking: {len(set(ra[ra<=K].index)&set(rb[rb<=K].index))}/{K}')
from scipy.stats import spearmanr
print(f'    Spearman(rank) = {spearmanr(ra,rb).statistic:.4f}')
ch=(cur[EVC].values!=base[EVC].values).any(1)
print(f'  evidence rows changed vs r13: {int(ch.sum())}  by family {cur.loc[ch,"predicted_behavior"].value_counts().to_dict()}')
PATCHES=[('r4/patch_r4_ci_stack_b3.csv','coordinated_isolation'),('r5/patch_r5_sp_zoo_w035.csv','soft_play'),
         ('r5/patch_r5_dt_zoo_gb15.csv','directed_transfer'),('r4/patch_r4_dt_typed_b1.csv','directed_transfer'),
         ('r4/patch_r4_f4_hybrid_tau03.csv','other_coordination')]
covered=set()
for rel,fam in PATCHES:
    p=f'{O}/{rel}'
    if not os.path.exists(p): print(f'  [missing] {rel}'); continue
    q=pd.read_csv(p,dtype=str,keep_default_na=False).set_index('pair_id')
    ok=[x for x in q.index if x in cur.index and cur.loc[x,'predicted_behavior']==fam]
    exact=int(sum((cur.loc[x,EVC].values==q.loc[x,EVC].values).all() for x in ok))
    print(f'  {rel:38s} {fam:22s} rows {len(q):5d} routed {len(ok):5d} | cell-exact in candidate {exact:5d}')
    if exact==len(ok): covered|=set(ok)
out=set(cur.index[ch])-covered
print(f'  changed rows NOT explained by any cell-exact patch: {len(out)}')
if out:
    print('   their families:', cur.loc[list(out),'predicted_behavior'].value_counts().to_dict())
