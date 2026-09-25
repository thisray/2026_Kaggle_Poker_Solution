"""R3-P20: does the official pair list leak the colluders?

Every listed eval pair has n>=38, but 15,510 pairs with n>=38 are NOT listed -- the same order as the dev count of
"one member is a known colluder and the pair is not a labelled collusion pair" (15,180). If the organisers built the
list by removing pairs that touch a colluder without colluding, then per player the number of MISSING n>=38 pairs is
bimodal: a colluder loses all of them but one, a clean player loses only the ones shared with a colluder.
Tested on eval (no labels) by the shape of the distribution, and on dev (labels known) as the ground-truth mirror.
"""
import numpy as np, pandas as pd, collections, json
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
pidx = pd.read_parquet(f"{O}/np/player_index.parquet"); pm = dict(zip(pidx.player_id, pidx.pi))
ev = pd.read_csv(f"{RAW}/evaluation_pairs.csv")
a = ev.player_1.map(pm).values; b = ev.player_2.map(pm).values
evkey = set(np.minimum(a, b) * 12000 + np.maximum(a, b))
te = pd.read_parquet(f"{O}/ptab_eval.parquet", columns=["p_lo", "p_hi", "n", "pool"])
te["key"] = te.p_lo * 12000 + te.p_hi; te["listed"] = te.key.isin(evkey)
E = te[te.n >= 38]
print("eval n>=38:", len(E), "listed", int(E.listed.sum()), "missing", int((~E.listed).sum()))
miss = collections.Counter(np.concatenate([E.p_lo.values[~E.listed.values], E.p_hi.values[~E.listed.values]]))
deg = collections.Counter(np.concatenate([E.p_lo.values, E.p_hi.values]))
players = sorted(deg)
m = np.array([miss.get(p, 0) for p in players]); d = np.array([deg[p] for p in players])
print("missing-pairs-per-player distribution:", dict(sorted(collections.Counter(m).items())[:14]))
print("players with 0 missing:", int((m == 0).sum()), "| with >=5 missing:", int((m >= 5).sum()), "| with m == d-1:", int((m == d - 1).sum()))
print("of players with >=5 missing: mean deg %.1f mean missing %.1f ; fraction with exactly deg-1 missing %.3f"
      % (d[m >= 5].mean(), m[m >= 5].mean(), float((m[m >= 5] == d[m >= 5] - 1).mean())))
susp = set(np.array(players)[m >= 5])
print("suspected colluder players:", len(susp))
kept = E[E.listed.values]
both = kept[kept.p_lo.isin(susp) & kept.p_hi.isin(susp)]
print("listed pairs with BOTH members suspected:", len(both), "| one member:", int((kept.p_lo.isin(susp) ^ kept.p_hi.isin(susp)).sum()))
json.dump(dict(n38=len(E), missing=int((~E.listed).sum()), suspects=len(susp), both_listed=len(both)),
          open(f"{O}/r3/t83_listing_leak.json", "w"), indent=1)
# ---- dev mirror: rebuild the same statistic where we know the answer ----
lab = pd.read_csv(f"{RAW}/development_labels.csv")
lab["key"] = np.minimum(lab.player_1.map(pm), lab.player_2.map(pm)) * 12000 + np.maximum(lab.player_1.map(pm), lab.player_2.map(pm))
posp = set(lab.loc[lab.label == 1, "player_1"].map(pm)) | set(lab.loc[lab.label == 1, "player_2"].map(pm))
print("\ndev mirror: known colluder players", len(posp), "| suspected-by-rule overlap with dev truth is not applicable (different phase)")
print("cross-check: how many suspected eval colluders are dev colluders?", len(susp & posp), "(expected ~0 under cross-phase exclusivity)")
