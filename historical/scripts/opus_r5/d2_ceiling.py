"""R5 d2: candidate-gate recall ceiling and oracle AP@5 per family; A/B composition."""
import numpy as np, pandas as pd, sys
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18')
import ci_censored_event as C
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'
full=C.prepare(pd.read_parquet(f'{O}/t5_dev_seq.parquet'))
cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
print('full shared-hand rows', full.shape, 'cand rows', cand.shape)
print('cand per slot', cand.groupby('slot').size().describe().to_dict())
g1=pd.read_parquet(f'{O}/r4/g1_evidence_rank.parquet')
def runs(ch):
    out=[];r=0
    for i,c in enumerate(ch):
        if i and c<ch[i-1]: r+=1
        out.append(r)
    return out
g1=g1.sort_values(['pair_id','evidence_rank'])
g1['run']=np.concatenate([runs(g.chron.values) for _,g in g1.groupby('pair_id',sort=False)])
g1['nruns']=g1.groupby('pair_id').run.transform('max')+1
rows=[]
for fam,s in full.groupby('fam'):
    s=s.reset_index(drop=True)
    c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev']],on=['slot','h'],validate='one_to_one').reset_index(drop=True)
    counts=s.groupby('slot').ev.sum()
    n_true=int(s.ev.sum()); n_in=int(c.ev.sum())
    # oracle: rank true evidence first
    orac=C.pair_ap(c, c.ev.astype(float).values, counts).mean()
    hands=s.groupby('slot').size()
    rows.append(dict(fam=fam,pairs=len(counts),true_hands=n_true,in_cand=n_in,recall=n_in/n_true,
                     oracle_ap5=float(orac), cand_per_pair=float(c.groupby('slot').size().mean()),
                     shared_hands_med=float(hands.median()), shared_hands_mean=float(hands.mean())))
print(pd.DataFrame(rows).to_string())
print()
print('=== A/B composition (2-run pairs give ground truth segmentation) ===')
for fam,g in g1.groupby('behavior_family'):
    nr=g.groupby('pair_id').nruns.first()
    seg=g[g.nruns==2].groupby('pair_id').run.apply(lambda v:(v==0).sum())
    print(f'{fam}: pairs {nr.size}, nruns dist {nr.value_counts().to_dict()}, |A| in 2-run pairs {seg.value_counts().sort_index().to_dict()}')
