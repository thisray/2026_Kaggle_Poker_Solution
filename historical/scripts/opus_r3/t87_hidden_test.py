"""R3-P24: are the suspected hidden dev positives real collusion, or just a persistent playing style?

Independent discriminator that uses no dev label: score the SAME player pair on the EVAL phase.
  * A real dev colluding pair stops colluding in eval (cross-phase exclusivity) -> its eval-phase score is ordinary.
  * A false positive is two players whose style merely looks collusive -> the style persists, so the eval-phase score
    stays high.
Groups: labelled dev positives (known real), the `hid` suspects (ref oof > 0.3), labelled dev negatives, and all
unlabelled pairs as the reference. Eval-phase percentiles are computed inside the pair's own pool to remove pool
effects, over ALL 174,000 within-pool pairs (not just the official list).
"""
import numpy as np, pandas as pd, json, os
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
pidx = pd.read_parquet(f"{O}/np/player_index.parquet"); pm = dict(zip(pidx.player_id, pidx.pi))
lab = pd.read_csv(f"{RAW}/development_labels.csv")
lab["key"] = np.minimum(lab.player_1.map(pm), lab.player_2.map(pm)) * 12000 + np.maximum(lab.player_1.map(pm), lab.player_2.map(pm))
TAG = "m15_o_pos_a"
es = pd.read_parquet(f"{O}/{TAG}_eval_scores.parquet")
print("eval score rows:", len(es), "columns:", list(es.columns))
es["key"] = es.p_lo * 12000 + es.p_hi
es["pct"] = es.groupby("pool").score.rank(pct=True)
ref = pd.read_parquet(f"{O}/m5_both_train_oof.parquet")
hid = {nm: set(ref[(ref.src == nm) & (ref.label == -1) & (ref.oof > 0.3)].key) for nm in ("devsub11", "devsub12")}
HID = hid["devsub11"] | hid["devsub12"]; HIDB = hid["devsub11"] & hid["devsub12"]
pos = set(lab.key[lab.label == 1]); neg = set(lab.key[lab.label == 0])
S = es.set_index("key")
def prof(name, keys):
    k = [x for x in keys if x in S.index]
    if not k: print(f"  {name}: none present"); return
    v = S.pct.loc[k].values
    print(f"  {name:28s} n={len(k):5d}  eval-phase within-pool percentile: median {np.median(v):.3f} mean {v.mean():.3f} "
          f"| share above 0.95: {np.mean(v > 0.95):.3f} | above 0.99: {np.mean(v > 0.99):.3f}")
    return float(np.median(v)), float(np.mean(v > 0.99))
print("\nEVAL-PHASE score of the same pairs (dev-trained model, no dev label used):")
prof("labelled dev POSITIVES", pos); prof("suspected hidden (either sub)", HID); prof("suspected hidden (both subs)", HIDB)
prof("labelled dev negatives", neg); prof("all pairs (reference)", list(S.index[::40]))
# and the dev-phase percentile for contrast, so the two phases are comparable
do = pd.read_parquet(f"{O}/{TAG}_train_oof.parquet"); do = do[do.src == "devsub11"]
D = do.assign(pct=do.groupby("pool").oof.rank(pct=True)).set_index("key")
print("\nDEV-PHASE percentile of the same groups (for contrast):")
for nm, ks in (("labelled dev POSITIVES", pos), ("suspected hidden (either)", HID), ("labelled dev negatives", neg)):
    k = [x for x in ks if x in D.index]
    if k: v = D.pct.loc[k].values; print(f"  {nm:28s} n={len(k):5d}  median {np.median(v):.3f}  above 0.99: {np.mean(v > 0.99):.3f}")
