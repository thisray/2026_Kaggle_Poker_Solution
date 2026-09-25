"""Round-11 scoped deployment gate builder (GB10, read-only on artifacts).

Outputs into OUT:
  eval_risk_with_slot.csv        slot,pair_id,risk_score   (all 112,540 eval pairs)
  slot_pair_id.csv               slot,pair_id
  gated_candidates.csv           candidates for selected slots (11 scores + ids)
  raw_shared_eval_membership.csv pair_id,hand_id,phase (phase verified from arrays)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OP = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
R8 = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round8_raw_20260917"
OUT = Path("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round11_scoped")
OUT.mkdir(parents=True, exist_ok=True)
BUDGET = 4000

pidx = pd.read_parquet(f"{OP}/np/player_index.parquet")
pmap = dict(zip(pidx.player_id, pidx.pi))
loc = pd.read_parquet(f"{OP}/player_local_v1.parquet").set_index("player_gi")
evalp = pd.read_csv(f"{RAW}/evaluation_pairs.csv")
lo = np.minimum(evalp.player_1.map(pmap), evalp.player_2.map(pmap)).values
hi = np.maximum(evalp.player_1.map(pmap), evalp.player_2.map(pmap)).values
evalp["key"] = lo * 12000 + hi
evalp["slot"] = loc.pool.loc[lo].values * 900 + loc.local.loc[lo].values * 30 + loc.local.loc[hi].values
assert evalp.slot.nunique() == len(evalp) == 112540

risk = pd.read_parquet(f"{OP}/m15_v6_drop_m26_eval_scores.parquet")[["key", "score"]]
m = evalp.merge(risk, on="key", how="left", validate="one_to_one")
assert m.score.notna().all()
m["risk_score"] = m.score
m[["slot", "pair_id", "risk_score"]].to_csv(OUT / "eval_risk_with_slot.csv", index=False)
evalp[["slot", "pair_id"]].to_csv(OUT / "slot_pair_id.csv", index=False)

cutoff = m.risk_score.nlargest(BUDGET).min()
sel = m.loc[m.risk_score >= cutoff, ["slot", "pair_id", "risk_score"]].copy()
sel.to_csv(OUT / "selected_slots.csv", index=False)
print("gate slots", len(sel), "cutoff", round(float(cutoff), 6))

# candidates for gated slots
cand = pd.read_csv(f"{R8}/eval_candidates.csv")
gc = cand[cand.slot.isin(set(sel.slot))].copy()
gc = gc.merge(evalp[["slot", "pair_id"]], on="slot", how="left", validate="many_to_one")
gc.to_csv(OUT / "gated_candidates.csv", index=False)
print("gated candidate rows", len(gc), "pairs", gc.slot.nunique(), "cols", list(gc.columns))

# membership with raw phase+shared verification
sp = np.load(f"{OP}/np/s_player.npy", mmap_mode="r")
phase = np.load(f"{OP}/np/h_phase.npy", mmap_mode="r")
hidx = pd.read_parquet(f"{OP}/np/hand_index.parquet").set_index("hand_id")
h = hidx.hi.reindex(gc.hand_id).to_numpy()
assert not np.isnan(h).any()
h = h.astype(int)
assert (np.asarray(phase[h]) == 1).all(), "non-evaluation hand in gated candidates"
pa = gc.pair_player_lo.map(pmap).to_numpy()
pb = gc.pair_player_hi.map(pmap).to_numpy()
seats = np.asarray(sp[h])
shared = (seats == pa[:, None]).any(1) & (seats == pb[:, None]).any(1)
assert shared.all(), "non-shared hand in gated candidates"
mem = pd.DataFrame({"pair_id": gc.pair_id, "hand_id": gc.hand_id, "phase": "evaluation"})
mem = mem.drop_duplicates()
mem.to_csv(OUT / "raw_shared_eval_membership.csv", index=False)
print("membership rows", len(mem))
print(json.dumps({"gate_slots": len(sel), "candidate_rows": len(gc), "membership_rows": len(mem)}))
