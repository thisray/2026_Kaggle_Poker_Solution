"""R5-E7: is the per-pair oracle over the zoo (DT 0.84 vs mean 0.73) real signal or
the maximum of 14 noisy estimates? Pick the best config per pair on seeds {0,1} and
evaluate that choice on seed 2 (and the two other rotations). If the choice
generalises the lane is worth more work; if it collapses to the mean, it is noise."""
import numpy as np, pandas as pd, sys, os, json, glob
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18')
import ci_censored_event as C
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; R5=f'{O}/r5'
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9)))
cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
out={}
for fam,ab in (('directed_transfer','di'),('soft_play','so'),('coordinated_isolation','co')):
    s=pd.read_parquet(f'{R5}/L/rows_{ab}.parquet').reset_index(drop=True); counts=s.groupby('slot').ev.sum()
    c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev']],on=['slot','h'],validate='one_to_one').reset_index(drop=True)
    fmi=pd.MultiIndex.from_arrays([s.slot,s.h]); mi=pd.MultiIndex.from_arrays([c.slot,c.h]); to_c=lambda v: pd.Series(v,index=fmi).reindex(mi).values
    files=sorted(glob.glob(f'{R5}/L/{ab}__*.npy')); nms=[os.path.basename(f).split('__')[1][:-4] for f in files]
    Z={n:np.load(f) for n,f in zip(nms,files)}; NS=Z[nms[0]].shape[0]
    A={}  # (config, seed) -> per-pair AP series
    for n in nms:
        for si in range(NS): A[(n,si)]=C.pair_ap(c,to_c(Z[n][si][2]),counts)
    idx=counts.index; M=lambda si: pd.DataFrame({n:A[(n,si)].reindex(idx).values for n in nms},index=idx)
    mean_all=float(np.mean([M(si).mean(axis=1).mean() for si in range(NS)]))
    same_seed_oracle=float(np.mean([M(si).max(axis=1).mean() for si in range(NS)]))
    held=[]
    for te in range(NS):
        tr=[si for si in range(NS) if si!=te]
        pick=(M(tr[0])+M(tr[1])).idxmax(axis=1)                 # best config per pair on the other two seeds
        held.append(float(np.mean([M(te).loc[p,pick[p]] for p in idx])))
    # a fair reference: the single best config chosen on the training seeds, applied to the held-out seed
    glob_=[]
    for te in range(NS):
        tr=[si for si in range(NS) if si!=te]
        best=(M(tr[0]).mean()+M(tr[1]).mean()).idxmax(); glob_.append(float(M(te)[best].mean()))
    out[fam]=dict(mean_of_configs=round(mean_all,4),same_seed_per_pair_oracle=round(same_seed_oracle,4),
                  cross_seed_per_pair_choice=round(float(np.mean(held)),4),cross_seed_single_best_config=round(float(np.mean(glob_)),4))
    print(fam,out[fam],flush=True)
json.dump(out,open(f'{R5}/e7_oracle_real.json','w'),indent=1)
