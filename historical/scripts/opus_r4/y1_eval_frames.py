"""R4-Y1: eval-side frames for a family (generalises r18_chatgpt/extract_ci_eval_gb10.py main(), which is CI-only):
all co-seated eval hands of the pairs routed to FAMILY in BASE, with the R18 gameplay columns, plus the frozen R15 top-20 candidates.
Also checks that BASE's evidence for those pairs is exactly the R15 top-5 (so the blend partner is the deployed ranking)."""
import numpy as np, pandas as pd, sys, json
from pathlib import Path
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18")
from extract_ci_eval_gb10 import extract
ROOT = Path("/home/thisray/projects/260916_Kaggle_Poker_artifacts"); OD = ROOT / "opus_r1_20260917"; D = OD / "np"
fam = sys.argv[1]; base_path = sys.argv[2]; tag = {"directed_transfer": "dt", "soft_play": "sp", "coordinated_isolation": "ci", "other_coordination": "f4"}[fam]
base = pd.read_csv(base_path, dtype=str, keep_default_na=False); sc = pd.read_csv(ROOT / "round15_campaign/scored_tabicl_rank_blend.csv")
ids = set(base.loc[base.predicted_behavior == fam, "pair_id"]); sc = sc[sc.pair_id.isin(ids)].copy()
hmap = pd.read_parquet(D / "hand_index.parquet").set_index("hand_id").hi; sc["h"] = sc.hand_id.map(hmap).astype("int64")
sc = sc.sort_values(["slot", "score"], ascending=[True, False], kind="stable").groupby("slot", sort=False).head(20).copy(); sc["r"] = sc.groupby("slot", sort=False).cumcount() + 1
print(f"{fam}: routed pairs {len(ids)}, with R15 candidates {sc.pair_id.nunique()}, candidates/pair min {int(sc.groupby('slot').size().min())}", flush=True)
EVC = [f"evidence_hand_{i}" for i in range(1, 6)]; b = base.set_index("pair_id")
top5 = sc[sc.r <= 5].groupby("pair_id").hand_id.apply(list)
same_set = np.mean([set(top5[p]) == set(b.loc[p, EVC]) for p in top5.index]); same_order = np.mean([list(top5[p]) == list(b.loc[p, EVC]) for p in top5.index])
print(f"BASE evidence == R15 top-5: same set {same_set:.4f}, same order {same_order:.4f}", flush=True)
loc = pd.read_parquet(OD / "player_local_v1.parquet"); mem = np.full((int(loc.pool.max()) + 1, 30), -1, np.int64); mem[loc.pool.values, loc.local.values] = loc.player_gi.values
sp = np.load(D / "s_player.npy", mmap_mode="r"); ht = np.load(D / "h_table.npy"); ph = np.load(D / "h_phase.npy"); keyed = []
for pool, pg in sc[["slot", "pair_id"]].drop_duplicates().assign(pool=lambda x: x.slot // 900).groupby("pool"):
    hs = np.flatnonzero((ht == pool) & (ph == 1)); seats = np.asarray(sp[hs])
    for row in pg.itertuples(index=False):
        sl = int(row.slot); pa = mem[pool, (sl % 900) // 30]; pb = mem[pool, sl % 30]; use = (seats == pa).any(axis=1) & (seats == pb).any(axis=1)
        keyed.append(pd.DataFrame({"slot": sl, "h": hs[use], "pair_id": row.pair_id, "fam": fam, "pa": min(pa, pb), "pb": max(pa, pb)}))
K = pd.concat(keyed, ignore_index=True); full = extract(OD, K); full.to_parquet(OD / f"r4/y1_{tag}_eval_full.parquet", index=False)
sc[["pair_id", "slot", "h", "hand_id", "r", "score"]].to_parquet(OD / f"r4/y1_{tag}_eval_candidates.parquet", index=False)
print(json.dumps({"pairs": int(sc.slot.nunique()), "shared_hands": len(full), "candidates": len(sc), "same_set": float(same_set)}))
