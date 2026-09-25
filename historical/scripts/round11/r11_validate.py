"""Round-11: strict validation of r11_scoped submission + dev gate OOF evaluation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

OP = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
S = Path("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round11_scoped")

sub = pd.read_csv(S / "r11_scoped.csv", dtype=str)
samp = pd.read_csv(f"{RAW}/sample_submission.csv", usecols=["pair_id"])
EVID = [f"evidence_hand_{i}" for i in range(1, 6)]
checks = {}
checks["rows"] = len(sub)
checks["unique_pair_id"] = int(sub.pair_id.nunique())
checks["pair_set_equal_sample"] = bool(set(sub.pair_id) == set(samp.pair_id))
sub["risk_score"] = sub.risk_score.astype(float)
checks["risk_in_0_1"] = bool(sub.risk_score.between(0, 1).all())
bad_dup = sum(len(set(r) - {"NO_EVIDENCE"}) != len([x for x in r if x != "NO_EVIDENCE"]) for r in sub[EVID].to_numpy())
checks["duplicate_evidence_in_pair"] = int(bad_dup)
checks["no_evidence_cells"] = int((sub[EVID].to_numpy() == "NO_EVIDENCE").sum())
checks["predicted_behavior_values"] = sub.predicted_behavior.value_counts().to_dict()

# phase + shared membership for all non-NO_EVIDENCE cells
sp = np.load(f"{OP}/np/s_player.npy", mmap_mode="r")
phase = np.load(f"{OP}/np/h_phase.npy", mmap_mode="r")
hidx = pd.read_parquet(f"{OP}/np/hand_index.parquet").set_index("hand_id")
pidx = pd.read_parquet(f"{OP}/np/player_index.parquet")
pmap = dict(zip(pidx.player_id, pidx.pi))
ev = pd.read_csv(f"{RAW}/evaluation_pairs.csv")
ev["lo"] = ev[["player_1", "player_2"]].min(axis=1).map(pmap)
ev["hi"] = ev[["player_1", "player_2"]].max(axis=1).map(pmap)
evmap = ev.set_index("pair_id")[["lo", "hi"]]
cells = sub[EVID].to_numpy().ravel()
cells = cells[cells != "NO_EVIDENCE"]
h = hidx.hi.reindex(pd.Index(cells)).to_numpy()
checks["all_cells_known_hands"] = bool(not np.isnan(h).any())
h = h.astype(int)
checks["all_cells_evaluation_phase"] = bool((np.asarray(phase[h]) == 1).all())
cell_mask = (sub[EVID].to_numpy().ravel() != "NO_EVIDENCE")
pair_ids = np.repeat(sub.pair_id.to_numpy(), 5)[cell_mask]
lo = evmap.lo.reindex(pair_ids).to_numpy(); hi = evmap.hi.reindex(pair_ids).to_numpy()
seats = np.asarray(sp[h])
checks["all_cells_shared"] = bool(((seats == lo[:, None]).any(1) & (seats == hi[:, None]).any(1)).all())
print(json.dumps(checks, indent=2))

# ---- dev gate OOF evaluation
oof = pd.read_parquet(f"{OP}/m15_v6_drop_m26_train_oof.parquet")
dev = oof[oof.src.isin(["devsub11", "devsub12"])][["key", "pool", "fold", "y", "oof"]].drop_duplicates("key")
pos = dev[dev.y == 1].copy()
pos["risk_rank"] = pos.oof.rank(ascending=False, method="average")
gate_recall = {}
for K in [2000, 4000, 8000]:
    gate_recall[K] = float((pos.risk_rank <= K).mean())
print("dev gate recall of labelled positives (rank<=K among 244k devsub)", gate_recall)
print("positives", len(pos), "median rank", float(pos.risk_rank.median()), "max rank", float(pos.risk_rank.max()))

# ranker blend in-gate E on labelled pairs
oof_s = pd.read_csv(S / "ranker_simple" / "oof_scores.csv.gz")
bp = oof_s.pivot_table(index="slot", values="blend", aggfunc="first") if "blend" in oof_s else None
m = pd.read_csv(f"/home/thisray/projects/260916_Kaggle_Poker_artifacts/round8_raw_20260917/dev_pack/meta.csv")
m = m.merge(oof_s[["slot", "hand_id", "blend"]], on=["slot", "hand_id"], how="left")
assert m.blend.notna().all()


def ap5(g, col):
    y = g.ev.to_numpy(int); mp = int(g.m_p.iloc[0])
    order = np.argsort(-g[col].to_numpy(), kind="stable"); top = y[order[:5]]
    return float(np.sum(top * np.cumsum(top) / np.arange(1, len(top) + 1)) / min(mp, 5))


rows = []
for slot, g in m.groupby("slot"):
    rows.append({"slot": slot, "pool": g.pool.iloc[0], "E_ranker": ap5(g, "blend")})
pp = pd.DataFrame(rows)
print("ranker_simple blend dev E (all 372 pairs)", round(float(pp.E_ranker.mean()), 6))
print(json.dumps({"validation": checks, "gate_recall_labelled_positives": gate_recall,
                  "ranker_blend_dev_E": float(pp.E_ranker.mean())}, indent=2))
json.dump({"validation": checks, "gate_recall": gate_recall, "ranker_blend_dev_E": float(pp.E_ranker.mean())},
          open(S / "r11_validate.json", "w"), indent=2)
