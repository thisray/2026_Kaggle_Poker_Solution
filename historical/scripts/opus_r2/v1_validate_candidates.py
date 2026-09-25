"""Independent legality check of candidate CSVs against raw data: columns, row set, risk range, behaviour labels,
evidence hands exist, are in the evaluation phase, include both players, no duplicates within a row."""
import sys, numpy as np, pandas as pd, hashlib
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; RAW = f"{A_}/data/raw"; C = f"{A_}/opus_r1_20260917/r2_candidates"
ss = pd.read_csv(f"{RAW}/sample_submission.csv", nrows=3); cols = list(ss.columns)
ev = pd.read_csv(f"{RAW}/evaluation_pairs.csv")
H = pd.read_parquet(f"{RAW}/hands.parquet", columns=["hand_id", "phase"]); evh = set(H[H.phase == "evaluation"].hand_id)
S = pd.read_parquet(f"{RAW}/seats.parquet", columns=["hand_id", "player_id"])
S = S[S.hand_id.isin(evh)]; seat = S.groupby("hand_id").player_id.apply(frozenset).to_dict()
pcols = [c for c in ev.columns if c.startswith("player")] or ["player_a", "player_b"]
print("eval pair columns:", list(ev.columns)[:6])
EVC = [f"evidence_hand_{i}" for i in range(1, 6)]
OK = {"none", "directed_transfer", "soft_play", "coordinated_isolation", "other_coordination"}
for f in sys.argv[1:]:
    d = pd.read_csv(f"{C}/{f}", dtype=str); bad = []
    if list(d.columns) != cols: bad.append("columns")
    if len(d) != len(ev) or set(d.pair_id) != set(ev.pair_id.astype(str)) or d.pair_id.duplicated().any(): bad.append("row set")
    r = d.risk_score.astype(float)
    if not (np.isfinite(r).all() and (r >= 0).all() and (r <= 1).all()): bad.append("risk")
    if not set(d.predicted_behavior) <= OK: bad.append("behavior")
    m = d.merge(ev.astype(str), on="pair_id")
    pa, pb = pcols[0], pcols[1]
    nbad = 0; ndup = 0; nn = 0
    for row in m[["pair_id", pa, pb] + EVC].itertuples(index=False):
        hs = [h for h in row[3:] if h != "NO_EVIDENCE"]; nn += len(hs)
        if len(set(hs)) != len(hs): ndup += 1
        for h in hs:
            s = seat.get(h)
            if s is None or row[1] not in s or row[2] not in s: nbad += 1
    if nbad: bad.append(f"{nbad} illegal evidence hands")
    if ndup: bad.append(f"{ndup} rows with duplicate evidence")
    print(f"{f}: {'PASS' if not bad else 'FAIL ' + str(bad)} | rows {len(d)} | evidence hands {nn} | other_coordination {int((d.predicted_behavior == 'other_coordination').sum())} | sha256 {hashlib.sha256(open(f'{C}/{f}', 'rb').read()).hexdigest()[:16]}")
