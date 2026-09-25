"""R4-U3: are DT members' big pots partner-specific? member x partner vs member x outsider; direction split."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917")
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; RAW = f"{A_}/data/raw"
H, S, T, SL = PI.all_pair_hands(0)
sp = np.load(f"{D}/s_player.npy"); con = np.load(f"{D}/s_contrib.npy"); net = np.load(f"{D}/s_net.npy"); bb = np.load(f"{D}/h_bb.npy").astype(float)
pS = sp[H, S]; pT = sp[H, T]
cS = con[H, S] / bb[H]; cT = con[H, T] / bb[H]; nS = net[H, S] / bb[H]; nT = net[H, T] / bb[H]
pidx = pd.read_parquet(f"{D}/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
lab = pd.read_csv(f"{RAW}/development_labels.csv"); pos = lab[lab.label == 1]
d = pd.read_parquet(f"{OUT}/t5_dev_seq.parquet")[["sl", "h", "fam", "zone"]]
df = pd.DataFrame({"sl": SL, "h": H, "pS": pS, "pT": pT, "cS": cS, "cT": cT, "nS": nS, "nT": nT}).merge(d, on=["sl", "h"], how="left")
for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]:
    mem = set(pos[pos.behavior_family == fam].player_1.map(pmap)) | set(pos[pos.behavior_family == fam].player_2.map(pmap))
    x = df[df.pS.isin(mem) | df.pT.isin(mem)].copy()
    x["grp"] = np.where(x.fam == fam, "partner:" + x.zone.fillna(""), "outsider")
    for th in (20, 40):
        x["big"] = (x.cS >= th) & (x.cT >= th)
        print(fam, "th", th); print(x.groupby("grp").big.agg(["mean", "sum", "size"]).round(4).to_string())
