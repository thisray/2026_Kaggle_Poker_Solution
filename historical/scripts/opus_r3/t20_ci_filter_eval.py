"""Eval patch: for pairs predicted coordinated_isolation, re-rank the r15 candidate pool with the CI listing condition (first member
preflop action call/raise with nobody folded before; dev evidence recall 97.4%): violators go below all compliant hands.
Output: patch table pair_id -> 5 evidence hands (only pairs whose top-5 changes).  BASE supplies predicted_behavior."""
import numpy as np, pandas as pd, os
from numba import njit
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"; R3 = f"{OUT}/r3"
BASE = os.environ.get("BASE", "r2f_NDw_all_on_r2j2mB6.csv")
base = pd.read_csv(f"{C}/{BASE}", dtype=str)
sc = pd.read_csv(f"{A_}/round15_campaign/scored_tabicl_rank_blend.csv")
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); id2hi = dict(zip(hidx.hand_id, hidx.hi)); sc["h"] = sc.hand_id.map(id2hi)
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); Y = np.load(f"{OUT}/dec_Y.npy"); a_pa = np.load(f"{D}/a_players_active.npy")
sp = np.load(f"{D}/s_player.npy", mmap_mode="r")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
H = sc.h.values; sl = sc.slot.values; plo = mem[sl // 900, (sl % 900) // 30]; phi = mem[sl // 900, sl % 30]
spH = np.asarray(sp[H]); S = np.argmax(spH == plo[:, None], axis=1); T = np.argmax(spH == phi[:, None], axis=1)
assert ((spH[np.arange(len(H)), S] == plo) & (spH[np.arange(len(H)), T] == phi)).all()
@njit
def first(H, S, T, off, a_seat, a_st, Y, a_pa, out):
    for r in range(len(H)):
        h = H[r]; out[r, 0] = -1; out[r, 1] = 0
        for k in range(off[h], off[h + 1]):
            if a_st[k] != 0: break
            if a_seat[k] == S[r] or a_seat[k] == T[r]:
                out[r, 0] = Y[k]; out[r, 1] = a_pa[k]; break
O = np.zeros((len(H), 2), np.int64); first(H, S.astype(np.int64), T.astype(np.int64), off, a_seat, a_st, Y, a_pa, O)
sc["ok"] = np.isin(O[:, 0], [2, 3]) & (O[:, 1] == 6)
beh = base.set_index("pair_id").predicted_behavior
sc["beh"] = sc.pair_id.map(beh)
cur = base.set_index("pair_id")[[f"evidence_hand_{i}" for i in range(1, 6)]]
# sanity: for known-family pairs the current evidence equals top-5 by r15 score
chk = []
for pid, g in sc[sc.beh.isin(["directed_transfer", "soft_play", "coordinated_isolation"])].groupby("pair_id"):
    top = g.sort_values("score", ascending=False, kind="mergesort").hand_id.values[:5].tolist(); chk.append(set(top) == set(cur.loc[pid].tolist()))
print(f"sanity: r15 top-5 == current evidence for {np.mean(chk):.4f} of {len(chk)} known-family gated pairs")
patch = {}
for pid, g in sc[sc.beh == "coordinated_isolation"].groupby("pair_id"):
    g = g.assign(s2=g.score - 1e6 * (~g.ok)).sort_values("s2", ascending=False, kind="mergesort")
    new = g.hand_id.values[:5].tolist()
    if new != cur.loc[pid].tolist(): patch[pid] = new
print(f"CI-predicted gated pairs {sc[sc.beh == 'coordinated_isolation'].pair_id.nunique()}; patched {len(patch)}; compliant share of old picks {sc[(sc.beh == 'coordinated_isolation')].groupby('pair_id').apply(lambda g: g.sort_values('score', ascending=False).ok.values[:5].mean()).mean():.3f}")
pd.DataFrame([(k, *v) for k, v in patch.items()], columns=["pair_id"] + [f"evidence_hand_{i}" for i in range(1, 6)]).to_parquet(f"{R3}/t20_ci_patch.parquet")
