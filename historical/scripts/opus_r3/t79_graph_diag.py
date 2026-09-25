"""Diagnostic for t78's graph block: is the candidate universe really that sparse, and is the blend a bug?"""
import numpy as np, pandas as pd, collections
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
b = pd.read_parquet(f"{O}/m15_v6ens_base_train_oof.parquet")
print("src values:", b.src.value_counts().to_dict())
for src in ("devsub11",):
    M = b[b.src == src][["key", "pool", "y", "hid", "oof"]]
    print(f"{src}: rows {len(M)} positives {int(M.y.sum())} hid {int(M.hid.sum())} pools {M.pool.nunique()}")
    k = M.key.values.astype(np.int64); plo = k // 12000; phi = k % 12000
    print("plo range", plo.min(), plo.max(), "phi range", phi.min(), phi.max())
    cnt = collections.Counter(np.concatenate([plo, phi]))
    print("pairs per player: mean %.2f median %d max %d; distinct players %d" % (np.mean(list(cnt.values())), np.median(list(cnt.values())), max(cnt.values()), len(cnt)))
    pos = M.y.values == 1; pp = set(plo[pos]) | set(phi[pos])
    print("players in subsample positives:", len(pp))
    inv = np.array([(a in pp) or (bb in pp) for a, bb in zip(plo, phi)])
    print("pairs touching a positive player:", int(inv.sum()), "of which positive:", int((inv & pos).sum()))
    c2 = collections.Counter(np.concatenate([plo[pos], phi[pos]]))
    print("positive-pair multiplicity of players:", dict(sorted(collections.Counter(c2.values()).items())))
    linked = np.array([ (c2.get(a,0) - (1 if p else 0)) > 0 or (c2.get(bb,0) - (1 if p else 0)) > 0 for a, bb, p in zip(plo, phi, pos)])
    print("linked pairs:", int(linked.sum()), "positive among them:", int((linked & pos).sum()))
    print("honest lift:", round(float((linked & pos).sum()/max(linked.sum(),1)) / float(pos.mean()), 2))
