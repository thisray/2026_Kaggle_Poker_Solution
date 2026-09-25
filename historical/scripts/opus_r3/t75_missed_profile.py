"""R3-P17: what distinguishes the positives the 48-model fusion still ranks beyond 400 from the ones it finds?
Uses the fusion OOF, the dev evidence table (how many listed hands, when they happen) and pair-level aggregates."""
import numpy as np, pandas as pd, glob, os, json
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
names = [os.path.basename(f).replace("_train_oof.parquet", "") for f in sorted(glob.glob(f"{O}/m*_train_oof.parquet")) if os.path.exists(f.replace("_train_oof", "_eval_scores"))]
pidx = pd.read_parquet(f"{O}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
lab = pd.read_csv(f"{RAW}/development_labels.csv"); lab["key"] = np.minimum(lab.player_1.map(pmap), lab.player_2.map(pmap)) * 12000 + np.maximum(lab.player_1.map(pmap), lab.player_2.map(pmap))
evd = pd.read_csv(f"{RAW}/development_evidence.csv").groupby("pair_id").size().rename("n_ev")
lab = lab.merge(evd, left_on="pair_id", right_index=True, how="left")
out = {}
for src in ("devsub11", "devsub12"):
    base = pd.read_parquet(f"{O}/m15_v6ens_base_train_oof.parquet"); M = base[base.src == src][["key", "pool", "y", "hid", "fam", "n"]].set_index("key")
    for n in names:
        d = pd.read_parquet(f"{O}/{n}_train_oof.parquet"); d = d[d.src == src][["key", "oof"]].rename(columns={"oof": n}).set_index("key")
        if len(d) and d.index.is_unique: M = M.join(d, how="left")
    mods = [c for c in M.columns if c not in ("pool", "y", "hid", "fam", "n") and M[c].notna().mean() > 0.9]
    M["fz"] = np.nanmean(np.stack([((M[m] - M[m].mean()) / M[m].std()).values for m in mods]), 0)
    K = M[~M.hid.astype(bool) | (M.y == 1)].copy(); K["rk"] = K.fz.rank(ascending=False, method="first")
    pos = K[K.y == 1].join(lab.set_index("key")[["n_ev", "behavior_family"]], how="left")
    miss = pos[pos.rk > 400]; hit = pos[pos.rk <= 400]
    print(f"== {src}: positives {len(pos)}, missed (rank>400) {len(miss)}")
    print("   family mix  missed:", dict(miss.behavior_family.value_counts()), "| found:", dict(hit.behavior_family.value_counts()))
    print(f"   listed evidence hands  missed mean {miss.n_ev.mean():.2f} (<5: {int((miss.n_ev < 5).sum())}/{len(miss)}) | found mean {hit.n_ev.mean():.2f} (<5: {int((hit.n_ev < 5).sum())}/{len(hit)})")
    print(f"   shared hands n         missed mean {miss.n.mean():.0f} median {miss.n.median():.0f} | found mean {hit.n.mean():.0f} median {hit.n.median():.0f}")
    print(f"   evidence per 100 hands missed {100 * (miss.n_ev / miss.n).mean():.2f} | found {100 * (hit.n_ev / hit.n).mean():.2f}")
    print("   missed pairs:"); print(miss[["behavior_family", "n", "n_ev", "rk"]].sort_values("rk").to_string())
    out[src] = dict(missed=int(len(miss)), miss_fam=dict(miss.behavior_family.value_counts()),
                    miss_nev=float(miss.n_ev.mean()), hit_nev=float(hit.n_ev.mean()),
                    miss_rate=float(100 * (miss.n_ev / miss.n).mean()), hit_rate=float(100 * (hit.n_ev / hit.n).mean()))
json.dump(out, open(f"{O}/r3/t75_missed_profile.json", "w"), indent=1)
