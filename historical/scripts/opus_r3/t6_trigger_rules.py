"""Validate crisp listing conditions on ALL known-family evidence (dev): players active / outsiders alive at the first member
action, first member action type, position; recall on evidence vs pass rate on in-window non-evidence and on our wrong picks."""
import numpy as np, pandas as pd
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D = f"{OUT}/np"
M = pd.read_parquet(f"{OUT}/t5_dev_seq.parquet")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
sp = np.load(f"{D}/s_player.npy", mmap_mode="r"); btn = np.load(f"{D}/h_btn.npy")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
H = M.h.values; plo = mem[M.sl.values // 900, (M.sl.values % 900) // 30]; phi = mem[M.sl.values // 900, M.sl.values % 30]
spH = np.asarray(sp[H]); S = np.argmax(spH == plo[:, None], axis=1); T = np.argmax(spH == phi[:, None], axis=1)
pos = np.asarray(Pt[H, :, PN.index("pos")]).astype(float); ix = np.arange(len(M))
M["pos_first"] = np.where(M.who == 0, pos[ix, S], np.where(M.who == 1, pos[ix, T], -1))
M["n_dealt"] = (spH >= 0).sum(1)
W = pd.read_parquet(f"{OUT}/t4_wrong_vs_hit.parquet")[["slot", "h", "kind", "r"]].rename(columns={"slot": "sl"})
M = M.merge(W, on=["sl", "h"], how="left")
pd.set_option("display.width", 220)
lab = {-1: "none", 0: "fold", 1: "check", 2: "call", 3: "raise"}
for fam in ["coordinated_isolation", "directed_transfer", "soft_play"]:
    F = M[(M.fam == fam) & (M.zone != "post")]
    E = F[F.ev]; N = F[~F.ev]
    print(f"\n== {fam}: evidence {len(E)}, in-window non-ev {len(N)}")
    print("  first-member action (ev):", E.y1.map(lab).value_counts(normalize=True).round(3).to_dict())
    print("  first-member action (non):", N.y1.map(lab).value_counts(normalize=True).round(3).to_dict())
    print("  players active at first member action (ev):", E.pa_at_trig.value_counts(normalize=True).sort_index().round(3).to_dict())
    print("  players active at first member action (non):", N.pa_at_trig.value_counts(normalize=True).sort_index().round(3).to_dict())
    print("  n dealt (ev):", E.n_dealt.value_counts().sort_index().to_dict(), " (non):", N.n_dealt.value_counts().sort_index().to_dict())
    print("  pos of first member (ev):", E.pos_first.value_counts(normalize=True).sort_index().round(3).to_dict())
    print("  pos of first member (non):", N.pos_first.value_counts(normalize=True).sort_index().round(3).to_dict())
    for nm, c in [("pa_at_trig == n_dealt (nobody folded before first member action)", F.pa_at_trig == F.n_dealt),
                  ("y1 == raise", F.y1 == 3), ("y1 == raise & nobody folded before", (F.y1 == 3) & (F.pa_at_trig == F.n_dealt)),
                  ("y1 in (call,raise) & nobody folded before", (F.y1 >= 2) & (F.pa_at_trig == F.n_dealt)),
                  ("y1 != fold", F.y1 != 0)]:
        e = c[F.ev].mean(); n = c[~F.ev].mean(); wr = c[F.kind == "wrong_in"].mean(); hit = c[F.kind == "hit"].mean()
        prec = F.ev[c].mean()
        print(f"  {nm:62s} recall(ev) {e:.3f} | pass(non) {n:.3f} | pass(wrong_in) {wr:.3f} | pass(hit) {hit:.3f} | P(ev|cond) {prec:.3f}")
