"""Known-family E: what separates our WRONG top-5 picks (non-evidence hands interleaved in time with the evidence) from the
HIT picks (true evidence)?  Per family, standardized mean differences over every per-member (P_v1), directed pair (R_v1) and
hand-level feature; plus near-necessary binary conditions (recall on hits vs pass rate on wrong picks)."""
import numpy as np, pandas as pd
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; RAW = f"{A_}/data/raw"; D_ = f"{OUT}/np"
d = pd.read_parquet(f"{A_}/round11_scoped/dev_oof_aligned.parquet")[["slot", "hand_id", "ev", "rs_blend"]]
tab = pd.read_csv(f"{A_}/round15_campaign/tabicl_cv/predictions.csv.gz")[["slot", "hand_id", "score"]].rename(columns={"score": "tab"})
d = d.merge(tab, on=["slot", "hand_id"])
d["b"] = 0.6 * d.groupby("slot").tab.rank(pct=True) + 0.4 * d.groupby("slot").rs_blend.rank(pct=True)
d["r"] = d.groupby("slot").b.rank(ascending=False, method="first")
hidx = pd.read_parquet(f"{D_}/hand_index.parquet").set_index("hand_id").hi; d["h"] = hidx.loc[d.hand_id].values
M = pd.read_parquet(f"{OUT}/s66_dev_outcomes.parquet").rename(columns={"sl": "slot"})
d = d.merge(M.drop(columns=["ev"]), on=["slot", "h"], how="left")
evt = M[M.ev].groupby("slot").ts.agg(["min", "max"])
d["tmin"] = d.slot.map(evt["min"]); d["tmax"] = d.slot.map(evt["max"])
d["kind"] = np.where(d.ev == 1, np.where(d.r <= 5, "hit", "miss"), np.where(d.r <= 5, np.where(d.ts > d.tmax, "wrong_late", "wrong_in"), "other"))
print(d.kind.value_counts().to_dict())
# per-member / pair features
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
sp = np.load(f"{D_}/s_player.npy", mmap_mode="r"); H = d.h.values
plo = mem[d.slot.values // 900, (d.slot.values % 900) // 30]; phi = mem[d.slot.values // 900, d.slot.values % 30]
spH = np.asarray(sp[H]); sa = np.argmax(spH == plo[:, None], axis=1); sb = np.argmax(spH == phi[:, None], axis=1)
L = open(f"{OUT}/feature_names_v1.txt").read().split("\n"); RN = L[0][2:].split(","); PN = L[1][2:].split(",")
P = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); R = np.load(f"{OUT}/R_v1.npy", mmap_mode="r")
PH = np.asarray(P[H]).astype(float); RH = np.asarray(R[H]).astype(float); ix = np.arange(len(d))
F = {}
for j, c in enumerate(PN):
    x = PH[ix, sa, j]; y = PH[ix, sb, j]
    F[f"P_{c}_max"] = np.maximum(x, y); F[f"P_{c}_min"] = np.minimum(x, y)
for j, c in enumerate(RN):
    x = RH[ix, sa, sb, j]; y = RH[ix, sb, sa, j]
    F[f"R_{c}_sum"] = x + y; F[f"R_{c}_absdiff"] = np.abs(x - y)
# outsiders: seats not a/b that were dealt in (s_player>=0)
dealt = spH >= 0; outs = dealt.copy(); outs[ix, sa] = False; outs[ix, sb] = False
for j, c in enumerate(PN):
    if c in ("folded", "sd", "won", "net_bb", "contrib_bb", "n_aggr", "vpip", "last_street"):
        v = PH[:, :, j].copy(); v[~outs] = np.nan
        F[f"O_{c}_sum"] = np.nansum(v, 1); F[f"O_{c}_max"] = np.nanmax(np.where(outs, PH[:, :, j], -1e9), 1)
F["n_dealt"] = dealt.sum(1); F["n_out"] = outs.sum(1)
X = pd.DataFrame(F); X["kind"] = d.kind.values; X["fam"] = d.fam.values
pd.set_option("display.width", 220)
for fam, G in X.groupby("fam"):
    hit = G[G.kind == "hit"]; wr = G[G.kind == "wrong_in"]
    rows = []
    for c in X.columns[:-2]:
        a, b = hit[c].astype(float), wr[c].astype(float)
        sd = np.sqrt((a.var() + b.var()) / 2) + 1e-9
        rows.append((c, a.mean(), b.mean(), (a.mean() - b.mean()) / sd))
    T = pd.DataFrame(rows, columns=["feat", "hit", "wrong_in", "smd"]).sort_values("smd", key=np.abs, ascending=False)
    print(f"== {fam}: hits {len(hit)} wrong_in {len(wr)}"); print(T.head(22).round(3).to_string(index=False))
    # binary near-necessary conditions: fraction of hits vs wrong passing x>0
    bins = []
    for c in X.columns[:-2]:
        a = (hit[c] > 0).mean(); b = (wr[c] > 0).mean()
        if a > 0.95 and b < 0.8: bins.append((c, "x>0", a, b))
        a0 = (hit[c] <= 0).mean(); b0 = (wr[c] <= 0).mean()
        if a0 > 0.95 and b0 < 0.8: bins.append((c, "x<=0", a0, b0))
    print("  near-necessary (hit recall>0.95, wrong pass<0.8):", [(c, k, round(a, 3), round(b, 3)) for c, k, a, b in bins][:20])
X.assign(slot=d.slot.values, h=d.h.values, r=d.r.values, ev=d.ev.values).to_parquet(f"{OUT}/t4_wrong_vs_hit.parquet")
