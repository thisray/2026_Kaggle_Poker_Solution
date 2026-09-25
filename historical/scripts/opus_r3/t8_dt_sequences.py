"""Print raw action sequences of DT / SP evidence hands (roles: A=p_lo member, B=p_hi member, o=outsider) to read the scripts;
tabulate the member action-pattern signature (first two member actions per street) for evidence vs in-window non-evidence."""
import numpy as np, pandas as pd, sys
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D = f"{OUT}/np"
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); Y = np.load(f"{OUT}/dec_Y.npy")
a_amt = np.load(f"{D}/a_amount.npy"); bbA = np.load(f"{D}/h_bb.npy")
sp = np.load(f"{D}/s_player.npy", mmap_mode="r"); snet = np.load(f"{D}/s_net.npy", mmap_mode="r")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
M = pd.read_parquet(f"{OUT}/t5_dev_seq.parquet")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
AC = "fxcr"
def sig(h, sa, sb, pf):
    s = []; parts = []
    for k in range(off[h], off[h + 1]):
        role = "A" if a_seat[k] == sa else ("B" if a_seat[k] == sb else "o")
        parts.append(f"{'PFTR'[a_st[k]]}{role}{AC[Y[k]]}{'' if a_amt[k] == 0 else int(a_amt[k] / bbA[h])}")
    return " ".join(parts)
def member_sig(h, sa, sb):
    # compact: per street, sequence of member actions only with the order relative to each other
    out = []
    for st in range(4):
        z = [("A" if a_seat[k] == sa else "B") + AC[Y[k]] for k in range(off[h], off[h + 1]) if a_st[k] == st and (a_seat[k] == sa or a_seat[k] == sb)]
        if z: out.append("".join(z))
    return "|".join(out)
pd.set_option("display.width", 250)
rng = np.random.RandomState(0)
for fam in sys.argv[1:]:
    F = M[(M.fam == fam) & (M.zone != "post")].copy()
    H = F.h.values; plo = mem[F.sl.values // 900, (F.sl.values % 900) // 30]; phi = mem[F.sl.values // 900, F.sl.values % 30]
    spH = np.asarray(sp[H]); SA = np.argmax(spH == plo[:, None], axis=1); SB = np.argmax(spH == phi[:, None], axis=1)
    netH = np.asarray(snet[H]).astype(float); ix = np.arange(len(F))
    F["msig"] = [member_sig(h, a, b) for h, a, b in zip(H, SA, SB)]
    F["netA"] = netH[ix, SA] / bbA[H]; F["netB"] = netH[ix, SB] / bbA[H]
    print(f"\n==== {fam}: top member signatures, evidence vs in-window non-evidence (share)")
    se = F[F.ev].msig.value_counts(normalize=True).head(25); sn = F[~F.ev].msig.value_counts(normalize=True)
    T = pd.DataFrame({"ev": se, "non": sn.reindex(se.index).fillna(0)}); T["P(ev|sig)"] = [F[F.msig == s].ev.mean() for s in T.index]
    print(T.round(3).to_string())
    print("  net (bb) of A/B in evidence: winner-loser orientation:")
    E = F[F.ev]; print("   frac one member +, other -:", round(((E.netA > 0) & (E.netB < 0) | (E.netA < 0) & (E.netB > 0)).mean(), 3))
    for i in rng.choice(np.flatnonzero(F.ev.values), 12, replace=False):
        h = H[i]; print(f"   ev  h={h} netA={F.netA.iloc[i]:+.1f} netB={F.netB.iloc[i]:+.1f} pfA={Pt[h, SA[i], PN.index('pf_eq_rand')]:.2f} pfB={Pt[h, SB[i], PN.index('pf_eq_rand')]:.2f} :: {sig(h, SA[i], SB[i], None)}")
