"""R3-P22: recover the EXACT set of dev colluders from the official eval pair list.

Established in t84: of the 174,000 within-pool eval pairs, the official list is exactly {n >= 38} minus every pair
that touches a player who colluded in the dev phase (listed rate 0.991 / 0.000 / 0.000 for 0 / 1 / 2 known colluders).
So a player whose n>=38 eval pairs are ALL unlisted is a dev colluder -- including the ones the official label file
does not name. This recovers the hidden positives exactly instead of the `ref oof > 0.3` heuristic.
"""
import numpy as np, pandas as pd, collections, json
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
pidx = pd.read_parquet(f"{O}/np/player_index.parquet"); pm = dict(zip(pidx.player_id, pidx.pi))
ev = pd.read_csv(f"{RAW}/evaluation_pairs.csv")
a = ev.player_1.map(pm).values; b = ev.player_2.map(pm).values
evkey = set(np.minimum(a, b) * 12000 + np.maximum(a, b))
te = pd.read_parquet(f"{O}/ptab_eval.parquet", columns=["p_lo", "p_hi", "n", "pool"])
te["listed"] = (te.p_lo * 12000 + te.p_hi).isin(evkey)
E = te[te.n >= 38]
deg = collections.Counter(np.concatenate([E.p_lo.values, E.p_hi.values]))
miss = collections.Counter(np.concatenate([E.p_lo.values[~E.listed.values], E.p_hi.values[~E.listed.values]]))
lab = pd.read_csv(f"{RAW}/development_labels.csv")
lab["key"] = np.minimum(lab.player_1.map(pm), lab.player_2.map(pm)) * 12000 + np.maximum(lab.player_1.map(pm), lab.player_2.map(pm))
posp = set(lab.loc[lab.label == 1, "player_1"].map(pm)) | set(lab.loc[lab.label == 1, "player_2"].map(pm))
allmiss = {p for p in deg if miss.get(p, 0) == deg[p]}
print(f"players with >=1 n>=38 eval pair: {len(deg)}; all of them unlisted: {len(allmiss)}")
print(f"  labelled dev colluders among them: {len(allmiss & posp)} of {len(posp)}")
new = sorted(allmiss - posp)
print(f"  NOT in the label file (recovered hidden colluders): {len(new)}")
print("  their degree distribution:", dict(sorted(collections.Counter([deg[p] for p in new]).items())[:12]))
strong = [p for p in new if deg[p] >= 5]
print(f"  of those with degree >= 5 (safe): {len(strong)}")
missing_labelled = sorted(posp - allmiss)
for p in missing_labelled: print(f"  labelled colluder not fully excluded: {p} deg {deg.get(p,0)} miss {miss.get(p,0)}")
loc = pd.read_parquet(f"{O}/player_local_v1.parquet").set_index("player_gi")
S = pd.DataFrame(dict(p=strong)); S["pool"] = loc.pool.loc[S.p].values
byp = S.groupby("pool").p.apply(list)
print(f"\nrecovered colluders sit in {len(byp)} pools; pool sizes:", dict(sorted(collections.Counter(byp.map(len)).items())))
# candidate hidden pairs: same-pool pairs among recovered players, plus recovered x labelled in the same pool
lp = pd.DataFrame(dict(p=sorted(posp))); lp["pool"] = loc.pool.loc[lp.p].values
labpool = lp.groupby("pool").p.apply(list).to_dict()
cand = []
for pool, ps in byp.items():
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)): cand.append((min(ps[i], ps[j]), max(ps[i], ps[j]), "new-new"))
        for q in labpool.get(pool, []): cand.append((min(ps[i], q), max(ps[i], q), "new-labelled"))
C = pd.DataFrame(cand, columns=["p_lo", "p_hi", "kind"]); C["key"] = C.p_lo * 12000 + C.p_hi
known = set(lab.key[lab.label == 1])
C = C[~C.key.isin(known)]
print("candidate hidden collusion pairs:", len(C), C.kind.value_counts().to_dict())
C.to_parquet(f"{O}/r3/t85_hidden_candidates.parquet")
np.save(f"{O}/r3/t85_colluder_players.npy", np.array(sorted(allmiss)))
# sanity: how do these candidates score in the existing dev model, and how do they compare with the `hid` heuristic?
d = pd.read_parquet(f"{O}/m15_o_touch_a_train_oof.parquet")
for src in ("devsub11", "devsub12"):
    M = d[d.src == src].copy(); M["pct"] = M.oof.rank(ascending=False, pct=True)
    hit = M[M.key.isin(set(C.key))]
    ref = pd.read_parquet(f"{O}/m5_both_train_oof.parquet"); H = set(ref[(ref.src == src) & (ref.label == -1) & (ref.oof > 0.3)].key)
    print(f"  {src}: recovered candidates present {len(hit)}; median percentile {hit.pct.median():.5f}; "
          f"in top-1000 {int((hit.pct <= 1000/len(M)).sum())} | heuristic hid set {len(H)}, overlap with recovered {len(H & set(C.key))}")
