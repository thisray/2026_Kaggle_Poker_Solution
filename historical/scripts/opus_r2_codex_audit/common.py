"""Shared read-only data access and independent p2 primitives for the audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange


ARTIFACT_ROOT = Path("/home/thisray/projects/260916_Kaggle_Poker_artifacts")
OUT = ARTIFACT_ROOT / "opus_r1_20260917"
DATA = ARTIFACT_ROOT / "data" / "raw"
NP = OUT / "np"
AUDIT_ART = Path("/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r2_codex_audit_20260918")


def ensure_audit_artifact_dir() -> Path:
    """Create and return the worker-owned artifact directory."""
    AUDIT_ART.mkdir(parents=True, exist_ok=True)
    return AUDIT_ART


def load_core_arrays():
    """Load only the arrays needed by the audit, keeping large arrays memory-mapped."""
    off = np.load(NP / "a_off.npy", mmap_mode="r")
    a_seat = np.load(NP / "a_seat.npy", mmap_mode="r")
    a_st = np.load(NP / "a_st.npy", mmap_mode="r")
    a_act = np.load(NP / "a_act.npy", mmap_mode="r")
    y = np.load(OUT / "dec_Y.npy", mmap_mode="r")
    probs = np.load(OUT / "dec_probs_v2.npy", mmap_mode="r")
    p_table = np.load(OUT / "P_v1.npy", mmap_mode="r")
    p_names = Path(OUT / "feature_names_v1.txt").read_text().splitlines()[1][2:].split(",")
    pfeq = np.asarray(p_table[:, :, p_names.index("pf_eq_rand")])
    h_phase = np.load(NP / "h_phase.npy", mmap_mode="r")
    h_table = np.load(NP / "h_table.npy", mmap_mode="r")
    h_bb = np.load(NP / "h_bb.npy", mmap_mode="r")
    h_sb = np.load(NP / "h_sb.npy", mmap_mode="r")
    h_btn = np.load(NP / "h_btn.npy", mmap_mode="r")
    h_ts = np.load(NP / "h_ts.npy", mmap_mode="r")
    s_player = np.load(NP / "s_player.npy", mmap_mode="r")
    hand_index = pd.read_parquet(NP / "hand_index.parquet")
    player_index = pd.read_parquet(NP / "player_index.parquet")
    player_local = pd.read_parquet(OUT / "player_local_v1.parquet").sort_values("player_gi")
    local_by_gi = player_local.local.to_numpy(dtype=np.int64)
    return {
        "off": off,
        "a_seat": a_seat,
        "a_st": a_st,
        "a_act": a_act,
        "y": y,
        "probs": probs,
        "pfeq": pfeq,
        "h_phase": h_phase,
        "h_table": h_table,
        "h_bb": h_bb,
        "h_sb": h_sb,
        "h_btn": h_btn,
        "h_ts": h_ts,
        "s_player": s_player,
        "hand_index": hand_index,
        "player_index": player_index,
        "local_by_gi": local_by_gi,
    }


@njit(cache=True)
def first_preflop_indices(off, a_seat, a_st):
    """Return the first preflop action index for every hand and seat."""
    n_hands = len(off) - 1
    first = -np.ones((n_hands, 6), dtype=np.int64)
    for h in range(n_hands):
        for k in range(off[h], off[h + 1]):
            if a_st[k] != 0:
                break
            seat = a_seat[k]
            if first[h, seat] < 0:
                first[h, seat] = k
    return first


@njit(parallel=True, cache=True)
def enumerate_phase_pairs(hands, s_player, h_table, local_by_gi):
    """Enumerate unordered six-seat pairs with the canonical pool-local slot."""
    n = len(hands)
    pair_h = np.empty(n * 15, dtype=np.int64)
    pair_s = np.empty(n * 15, dtype=np.int8)
    pair_t = np.empty(n * 15, dtype=np.int8)
    pair_slot = np.empty(n * 15, dtype=np.int64)
    for r in prange(n):
        h = hands[r]
        q = 0
        for s in range(6):
            for t in range(s + 1, 6):
                gs = s_player[h, s]
                gt = s_player[h, t]
                ls = local_by_gi[gs]
                lt = local_by_gi[gt]
                row = r * 15 + q
                pair_h[row] = h
                if ls < lt:
                    pair_s[row] = s
                    pair_t[row] = t
                    lo = ls
                    hi = lt
                else:
                    pair_s[row] = t
                    pair_t[row] = s
                    lo = lt
                    hi = ls
                pair_slot[row] = h_table[h] * 900 + lo * 30 + hi
                q += 1
    return pair_h, pair_s, pair_t, pair_slot


def phase_pairs(data: dict, phase: int):
    hands = np.flatnonzero(np.asarray(data["h_phase"]) == phase).astype(np.int64)
    return enumerate_phase_pairs(hands, data["s_player"], data["h_table"], data["local_by_gi"])


@njit(cache=True)
def accumulate_first_preflop(pair_h, pair_s, pair_t, pair_slot, first, pfeq, mu, y, probs):
    """Accumulate the two directional Rao-score components for every pair slot."""
    acc = np.zeros((400 * 900, 2, 6), dtype=np.float64)
    for r in range(len(pair_h)):
        h = pair_h[r]
        for direction in range(2):
            actor = pair_s[r] if direction == 0 else pair_t[r]
            partner = pair_t[r] if direction == 0 else pair_s[r]
            actor_k = first[h, actor]
            partner_k = first[h, partner]
            if actor_k < 0:
                continue
            if partner_k >= 0 and partner_k < actor_k:
                continue
            e = pfeq[h, partner] - mu
            pf = probs[actor_k, 0]
            pa = probs[actor_k, 3]
            rf = (1.0 if y[actor_k] == 0 else 0.0) - pf
            ra = (1.0 if y[actor_k] == 3 else 0.0) - pa
            slot = pair_slot[r]
            acc[slot, direction, 0] += rf * e
            acc[slot, direction, 1] += pf * (1.0 - pf) * e * e
            acc[slot, direction, 2] += ra * e
            acc[slot, direction, 3] += pa * (1.0 - pa) * e * e
            acc[slot, direction, 4] += 1.0
            acc[slot, direction, 5] += e * e
    return acc


def score_frame(acc: np.ndarray) -> pd.DataFrame:
    zf = acc[:, :, 0] / np.sqrt(acc[:, :, 1] + 1e-9)
    za = acc[:, :, 2] / np.sqrt(acc[:, :, 3] + 1e-9)
    frame = pd.DataFrame(
        {
            "slot": np.arange(acc.shape[0], dtype=np.int64),
            "n0": acc[:, 0, 4],
            "n1": acc[:, 1, 4],
            "zf0": zf[:, 0],
            "zf1": zf[:, 1],
            "za0": za[:, 0],
            "za1": za[:, 1],
        }
    )
    frame = frame[(frame.n0 + frame.n1) > 0].copy()
    z = frame[["zf0", "zf1", "za0", "za1"]].to_numpy()
    frame["chi"] = (z * z).sum(axis=1)
    frame["zmax"] = np.abs(z).max(axis=1)
    frame["p2"] = np.minimum(frame.za0 - frame.zf0, frame.za1 - frame.zf1)
    frame["n_is"] = frame.n0 + frame.n1
    return frame


def compute_first_preflop_scores(data: dict, phase: int):
    first = first_preflop_indices(data["off"], data["a_seat"], data["a_st"])
    pair_h, pair_s, pair_t, pair_slot = phase_pairs(data, phase)
    mu = float(data["pfeq"].mean())
    acc = accumulate_first_preflop(
        pair_h,
        pair_s,
        pair_t,
        pair_slot,
        first,
        data["pfeq"],
        mu,
        data["y"],
        data["probs"],
    )
    return score_frame(acc), first, (pair_h, pair_s, pair_t, pair_slot), mu


def player_maps(data: dict):
    player_index = data["player_index"]
    pi_by_id = dict(zip(player_index.player_id, player_index.pi.astype(int)))
    id_by_pi = dict(zip(player_index.pi.astype(int), player_index.player_id))
    pl = pd.read_parquet(OUT / "player_local_v1.parquet")
    pool_by_pi = dict(zip(pl.player_gi.astype(int), pl.pool.astype(int)))
    local_by_pi = dict(zip(pl.player_gi.astype(int), pl.local.astype(int)))
    return pi_by_id, id_by_pi, pool_by_pi, local_by_pi


def pair_slot_from_players(pool_by_pi, local_by_pi, p1_pi: int, p2_pi: int) -> int:
    pool = int(pool_by_pi[p1_pi])
    lo = min(int(local_by_pi[p1_pi]), int(local_by_pi[p2_pi]))
    hi = max(int(local_by_pi[p1_pi]), int(local_by_pi[p2_pi]))
    return pool * 900 + lo * 30 + hi


def read_evaluation_pairs():
    return pd.read_csv(DATA / "evaluation_pairs.csv")


def read_candidate(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / "r2_candidates" / name, dtype=str)


def rank_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["risk_score", "pair_id"], ascending=[False, True]).reset_index(drop=True).copy()
    out["rank"] = np.arange(1, len(out) + 1)
    return out
