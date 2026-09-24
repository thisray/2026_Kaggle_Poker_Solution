"""Completion-conditioned dense variants (for the follow-up if the dense class wins): evidence = first 5 hands with an ACTIVE
decision AND a completed interaction.  Completion candidates: pair wins the pot (NDw); partner (the other member) wins (NDp);
both members stay to the flop (NDf).  Uses the extended per-hand table (c14) on the r2j2m base; reports overlaps."""
import os
import json
import numpy as np, pandas as pd, hashlib
OUT = os.environ["POKER_WORK_DIR"]; D = f"{OUT}/np"; C = OUT
H = pd.read_parquet(f"{OUT}/c14_hand_tables_ext.parquet").sort_values(["slot", "ts"]).reset_index(drop=True)
sp = np.load(f"{D}/s_player.npy", mmap_mode="r"); swon = np.load(f"{D}/s_won.npy", mmap_mode="r")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
hh = H.h.values; sl = H.slot.values
pa = mem[sl // 900, (sl % 900) // 30]; pb = mem[sl // 900, sl % 30]
spH = np.asarray(sp[hh]); sa = np.argmax(spH == pa[:, None], axis=1); sb = np.argmax(spH == pb[:, None], axis=1)
won = np.asarray(swon[hh]); ix = np.arange(len(H))
H["wa"] = won[ix, sa] > 0; H["wb"] = won[ix, sb] > 0; H["pair_win"] = H.wa | H.wb
ls = np.asarray(Pt[hh, :, PN.index("last_street")]); H["both_flop"] = (ls[ix, sa] >= 1) & (ls[ix, sb] >= 1)
print("share of member hands: pair wins %.3f, both see flop %.3f" % (H.pair_win.mean(), H.both_flop.mean()))
def first_k_prob(q, K=5):
    out = np.zeros(len(q)); dist = np.zeros(K + 1); dist[0] = 1.0
    for i, p in enumerate(q):
        out[i] = p * dist[:K].sum(); nd = dist * (1 - p); nd[1:] += dist[:-1] * p; nd[K] += dist[K] * p; dist = nd
    return out
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hi2id = dict(zip(hidx.hi, hidx.hand_id))
base = pd.read_csv(f"{C}/f4_routed_baseline.csv", dtype=str)
mids = base[base.predicted_behavior == "other_coordination"].pair_id.tolist()
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet"); pid2sl = c38.set_index("pair_id").slot
groups = {s: G for s, G in H.groupby("slot")}
for nm, cond in [("NDw", "pair_win")]:
    out = base.set_index("pair_id").copy(); pk = {}
    for pid in mids:
        s = pid2sl[pid]; G = groups[s]
        q = G.q_HD.values * G[cond].values.astype(float)
        sc = first_k_prob(q) + 1e-6 * first_k_prob(G.q_HD.values)       # tie-break / fill with plain dense order
        p = G.h.values[np.argsort(-sc, kind="stable")][:5].tolist(); assert len(set(p)) == 5
        out.loc[pid, [f"evidence_hand_{i}" for i in range(1, 6)]] = [hi2id[x] for x in p]; pk[s] = p
    o = out.reset_index()[base.columns]; path = f"{C}/f4_ndw_baseline.csv"; o.to_csv(path, index=False)
    receipt = {
        "pairs": len(mids),
        "evidence_changed_rows": int((o[[f"evidence_hand_{i}" for i in range(1, 6)]].to_numpy() != base[[f"evidence_hand_{i}" for i in range(1, 6)]].to_numpy()).any(axis=1).sum()),
        "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
    }
    with open(f"{C}/f4_ndw.receipt.json", "w") as handle:
        json.dump(receipt, handle, indent=2)
    print(json.dumps(receipt, indent=2))
