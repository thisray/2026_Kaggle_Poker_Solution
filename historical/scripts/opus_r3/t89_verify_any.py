"""R3-QA2: model-independent verification of ANY submission file.

Checks that need no knowledge of how the file was produced:
  1. exactly the official pair_id set, once each, and the official column layout;
  2. risk_score parses, is finite and inside [0, 1];
  3. predicted_behavior is one of the four allowed strings;
  4. every listed evidence hand exists, is distinct inside its row, sits in the EVAL phase of the pair's own pool, and
     BOTH members of the pair were actually seated in that hand -- a hand that fails this scores zero by construction;
  5. evidence rows are dense (no gap: a blank followed by a non-blank).
Usage: PY t89_verify_any.py <file.csv> [more.csv ...]
"""
import numpy as np, pandas as pd, sys, hashlib
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; O = f"{A_}/opus_r1_20260917"; RAW = f"{A_}/data/raw"; D = f"{O}/np"
ALLOWED = {"directed_transfer", "soft_play", "coordinated_isolation", "other_coordination"}
EVC = [f"evidence_hand_{i}" for i in range(1, 6)]
COLS = ["pair_id", "risk_score", "predicted_behavior"] + EVC
ev = pd.read_csv(f"{RAW}/evaluation_pairs.csv"); pidx = pd.read_parquet(f"{D}/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
ev["a"] = ev.player_1.map(pmap); ev["b"] = ev.player_2.map(pmap)
official = set(ev.pair_id); PA = dict(zip(ev.pair_id, ev.a)); PB = dict(zip(ev.pair_id, ev.b))
hidx = pd.read_parquet(f"{D}/hand_index.parquet").set_index("hand_id").hi
sp = np.load(f"{D}/s_player.npy", mmap_mode="r")
loc = pd.read_parquet(f"{O}/player_local_v1.parquet").set_index("player_gi")
for path in sys.argv[1:]:
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    name = path.split("/")[-1]; bad = []
    if list(d.columns) != COLS: bad.append(f"columns {list(d.columns)}")
    if len(d) != len(official): bad.append(f"rows {len(d)} != {len(official)}")
    if set(d.pair_id) != official: bad.append("pair_id set mismatch")
    if d.pair_id.duplicated().any(): bad.append("duplicate pair_id")
    r = pd.to_numeric(d.risk_score, errors="coerce")
    if r.isna().any() or not ((r >= 0) & (r <= 1)).all(): bad.append("risk_score out of [0,1] or unparseable")
    if not set(d.predicted_behavior.unique()) <= ALLOWED: bad.append(f"behavior values {set(d.predicted_behavior.unique()) - ALLOWED}")
    E = d[EVC].values
    blank = E == ""
    if (blank[:, :-1] & ~blank[:, 1:]).any(): bad.append("gap inside the evidence list")
    flat = E.ravel(); nz = flat != ""
    unknown = set(flat[nz]) - set(hidx.index)
    if unknown: bad.append(f"{len(unknown)} evidence hand ids not in the data")
    # row-wise distinctness
    dup = sum(1 for row in E if len(set(x for x in row if x)) != sum(1 for x in row if x))
    if dup: bad.append(f"{dup} rows repeat an evidence hand")
    # seating / phase check
    hi = pd.Series(flat[nz]).map(hidx).values.astype(np.int64)
    pid = np.repeat(d.pair_id.values, 5)[nz]
    pa = np.array([PA[p] for p in pid]); pb = np.array([PB[p] for p in pid])
    seats = np.asarray(sp[hi])
    both = ((seats == pa[:, None]).any(1)) & ((seats == pb[:, None]).any(1))
    phase_ok = (hi % 5000) >= 3000
    pool_ok = (hi // 5000) == loc.pool.loc[pa].values
    if not both.all(): bad.append(f"{int((~both).sum())} evidence hands where the two players were not both seated")
    if not phase_ok.all(): bad.append(f"{int((~phase_ok).sum())} evidence hands outside the eval phase")
    if not pool_ok.all(): bad.append(f"{int((~pool_ok).sum())} evidence hands from another pool")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    n_ev = int(nz.sum()); n5 = int((~blank).sum(1).__eq__(5).sum())
    print(f"{name:42s} sha {sha}  rows {len(d)}  evidence {n_ev} ({n5} pairs with 5)  "
          f"families {dict(d.predicted_behavior.value_counts())}")
    print("   " + ("PASS" if not bad else "FAIL: " + "; ".join(bad)), flush=True)
