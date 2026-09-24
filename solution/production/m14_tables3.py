import numpy as np, pandas as pd, time
import aggmod as AG
OUT = AG.OUT; t0 = time.time()
n4 = open(f"{OUT}/feature_names4_v2.txt").read().split("\n"); RN4 = n4[0][2:].split(","); PN4 = n4[1][2:].split(",")
R4 = np.load(f"{OUT}/R4_v2.npy"); P4 = np.load(f"{OUT}/P4_v2.npy")
for kind, seed, name in [("dev", 0, "dev"), ("eval", 0, "eval"), ("devsub", 11, "devsub11"), ("devsub", 12, "devsub12")]:
    t = AG.aggregate(R4, P4, RN4, PN4, AG.hand_mask(kind, seed), "Q_")
    t.to_parquet(f"{OUT}/ptab4_{name}.parquet"); print(name, t.shape, time.time() - t0, flush=True)
