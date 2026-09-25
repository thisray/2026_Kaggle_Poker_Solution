"""DT direction: is the evidence chip flow always donor->receiver with a fixed receiver per pair?  Among in-window hands whose
member signature is 'evidence-like' (one member raises, the other calls, then folds to the same member), are unlisted ones the
reverse direction?  And what are our wrong DT picks?"""
import numpy as np, pandas as pd
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D = f"{OUT}/np"
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); Y = np.load(f"{OUT}/dec_Y.npy")
bbA = np.load(f"{D}/h_bb.npy"); sp = np.load(f"{D}/s_player.npy", mmap_mode="r"); snet = np.load(f"{D}/s_net.npy", mmap_mode="r")
M = pd.read_parquet(f"{OUT}/t5_dev_seq.parquet")
W = pd.read_parquet(f"{OUT}/t4_wrong_vs_hit.parquet")[["slot", "h", "kind"]].rename(columns={"slot": "sl"})
M = M.merge(W, on=["sl", "h"], how="left")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
AC = "fxcr"
def msig(h, sa, sb):
    out = []
    for st in range(4):
        z = [("A" if a_seat[k] == sa else "B") + AC[Y[k]] for k in range(off[h], off[h + 1]) if a_st[k] == st and (a_seat[k] == sa or a_seat[k] == sb)]
        if z: out.append("".join(z))
    return "|".join(out)
for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]:
    F = M[(M.fam == fam)].copy()
    H = F.h.values; plo = mem[F.sl.values // 900, (F.sl.values % 900) // 30]; phi = mem[F.sl.values // 900, F.sl.values % 30]
    spH = np.asarray(sp[H]); SA = np.argmax(spH == plo[:, None], axis=1); SB = np.argmax(spH == phi[:, None], axis=1)
    netH = np.asarray(snet[H]).astype(float); ix = np.arange(len(F))
    F["netA"] = netH[ix, SA]; F["netB"] = netH[ix, SB]
    F["sig"] = [msig(h, a, b) for h, a, b in zip(H, SA, SB)]
    F["win"] = np.where(F.netA > F.netB, "A", "B")
    # receiver per pair = member who wins most evidence hands
    E = F[F.ev]; rec = E.groupby("sl").win.agg(lambda s: s.value_counts().index[0]); cons = E.groupby("sl").win.agg(lambda s: s.value_counts().iloc[0] / len(s))
    print(f"\n== {fam}: evidence winner consistency per pair: mean {cons.mean():.3f}; share of pairs 100% consistent {(cons == 1).mean():.3f}")
    F["rec"] = F.sl.map(rec); F["to_rec"] = F.win == F.rec
    Wn = F[F.zone != "post"]
    # evidence-like signature: member X raises and member Y calls in some street, later Y folds to X
    def dumplike(s):
        st = s.split("|")
        for i, a in enumerate(st):
            for x, y in (("A", "B"), ("B", "A")):
                if f"{x}r{y}c" in a and any(f"{x}r{y}f" in b for b in st[i + 1:] + [a[a.find(f'{x}r{y}c') + 4:]]): return x
        return ""
    Wn = Wn.assign(dl=[dumplike(s) for s in Wn.sig])
    D1 = Wn[Wn.dl != ""]
    print(f"  in-window dump-like hands: {len(D1)}; P(ev) {D1.ev.mean():.3f}; P(ev | winner==pair receiver) {D1[D1.to_rec].ev.mean():.3f} (n={int(D1.to_rec.sum())}); P(ev | winner!=receiver) {D1[~D1.to_rec].ev.mean():.3f} (n={int((~D1.to_rec).sum())})")
    print(f"  all in-window hands: P(ev | winner==receiver & one+ one-) {Wn[Wn.to_rec & (Wn.netA * Wn.netB < 0)].ev.mean():.3f}; P(ev | winner!=receiver) {Wn[~Wn.to_rec].ev.mean():.3f}")
    wr = F[F.kind == "wrong_in"]; hit = F[F.kind == "hit"]
    print(f"  our wrong_in picks {len(wr)}: winner==receiver {wr.to_rec.mean():.3f}; hits: {hit.to_rec.mean():.3f}")
    print(f"  wrong_in picks one member +/other -: {(wr.netA * wr.netB < 0).mean():.3f}; hits {(hit.netA * hit.netB < 0).mean():.3f}")
