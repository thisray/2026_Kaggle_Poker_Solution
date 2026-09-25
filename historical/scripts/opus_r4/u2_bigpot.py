"""R4-U2: base rate of big mutual pots between two co-seated players: positives (by family, ev / in-window non-ev / post) vs everyone else (dev phase)."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917")
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
H, S, T, SL = PI.all_pair_hands(0)
con = np.load(f"{D}/s_contrib.npy"); net = np.load(f"{D}/s_net.npy"); bb = np.load(f"{D}/h_bb.npy").astype(float)
cS = con[H, S] / bb[H]; cT = con[H, T] / bb[H]; nS = net[H, S] / bb[H]; nT = net[H, T] / bb[H]
big = (cS >= 40) & (cT >= 40)
d = pd.read_parquet(f"{OUT}/t5_dev_seq.parquet")[["sl", "h", "fam", "zone"]]
df = pd.DataFrame({"sl": SL, "h": H, "big": big, "cS": cS, "cT": cT, "nS": nS, "nT": nT}).merge(d, on=["sl", "h"], how="left")
df["fam"] = df.fam.fillna("rest"); df["zone"] = df.zone.fillna("-")
print(df.groupby(["fam", "zone"]).big.agg(["mean", "sum", "size"]).to_string())
for th in (10, 20, 40, 80):
    df["b"] = (df.cS >= th) & (df.cT >= th)
    print("th", th); print(df.groupby(["fam", "zone"]).b.mean().unstack().round(4).to_string())
