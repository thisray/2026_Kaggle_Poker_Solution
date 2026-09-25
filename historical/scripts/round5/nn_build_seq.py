"""Step B1 of r5b: build transformer input sequences for candidate rows (numba, opus env)."""
import numpy as np
import pandas as pd
import time
import seq_prep

OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
DST = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917"
t0 = time.time()
candy = pd.read_parquet(f"{DST}/r5_candidates.parquet").sort_values(["slot", "h"]).reset_index(drop=True)
print("candidates", len(candy), flush=True)
X, M, S = seq_prep.make(candy.h.values, candy.sa.values, candy.sb.values)
np.save(f"{DST}/r5cand_X.npy", X)
np.save(f"{DST}/r5cand_M.npy", M)
np.save(f"{DST}/r5cand_S.npy", S)
cols = ["slot", "h", "s1", "fi", "u0", "lin", "u"] + (["gen_rank_all"] if "gen_rank_all" in candy.columns else [])
candy[cols].to_parquet(f"{DST}/r5cand_meta.parquet")
print("done", X.shape, round(time.time() - t0, 1), flush=True)
