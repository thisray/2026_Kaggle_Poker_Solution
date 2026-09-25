"""R4-Z7a: eval frames (R18 gameplay columns) for the z1 comparison groups: random control pairs and top known-family pairs, to score them with the same two-orientation DT typed models as F4."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18"); sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917")
from extract_ci_eval_gb10 import extract
import pairindex as PI
from pathlib import Path
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
p = pd.read_parquet(f"{O}/r4/z1_pairs.parquet"); H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, p.slot.values)
K = pd.DataFrame({"slot": SL[m], "h": H[m]}); K["fam"] = "ctrl"; K["pair_id"] = K.slot.astype(str); K = K.merge(p[["slot", "pa", "pb"]], on="slot")
full = extract(Path(O), K); full.to_parquet(f"{O}/r4/z7_groups_eval_full.parquet", index=False); print("saved", full.shape, full.slot.nunique())
