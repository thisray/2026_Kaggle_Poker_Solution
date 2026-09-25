"""DT direction lever: share of r15 top-5 false positives whose pair-flow direction is opposite to the pair's true
donor->receiver direction (from evidence); oracle gain from demoting reverse-direction candidates; how well direction can
be inferred without labels (orientation features already in the pack: o_dir_agree, o_flow_dr, ...)."""
import numpy as np, pandas as pd
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D_ = f"{OUT}/np"
RN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[0][2:].split(",")
R = np.load(f"{OUT}/R_v1.npy", mmap_mode="r"); sp = np.load(f"{D_}/s_player.npy")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
d = pd.read_parquet(f"{A_}/round11_scoped/dev_oof_aligned.parquet")[["slot", "hand_id", "fold", "ev", "m_p", "rs_blend"]]
t = pd.read_csv(f"{A_}/round15_campaign/tabicl_cv/predictions.csv.gz")[["slot", "hand_id", "score"]].rename(columns={"score": "tab"}); d = d.merge(t, on=["slot", "hand_id"])
n = pd.read_csv(f"{A_}/round3_research_20260917/r6_narrow_candidates_v2.csv", usecols=["slot", "hand_id", "family", "hand_ts", "o_dir_agree", "o_flow_dr", "o_dir_conf", "z_o_dir_agree"]); d = d.merge(n, on=["slot", "hand_id"])
hidx = pd.read_parquet(f"{D_}/hand_index.parquet"); d["h"] = d.hand_id.map(dict(zip(hidx.hand_id, hidx.hi))).astype(np.int64)
d["r15"] = 0.6 * d.groupby("slot").tab.rank(pct=True) + 0.4 * d.groupby("slot").rs_blend.rank(pct=True)
plo = mem[d.slot.values // 900, (d.slot.values % 900) // 30]; phi = mem[d.slot.values // 900, d.slot.values % 30]
sa = np.argmax(sp[d.h.values] == plo[:, None], axis=1); sb = np.argmax(sp[d.h.values] == phi[:, None], axis=1)
fi = RN.index("flow"); d["f_ab"] = np.asarray(R[d.h.values, sa, sb, fi]); d["f_ba"] = np.asarray(R[d.h.values, sb, sa, fi])
d["dir"] = np.sign(d.f_ab - d.f_ba)            # +1: lo->hi value flow, -1: hi->lo, 0: none
dt = d[d.family == "directed_transfer"].copy()
truth = dt[dt.ev == 1].groupby("slot").dir.agg(lambda s: np.sign(s.sum()))
dt["tdir"] = dt.slot.map(truth)
dt["rank"] = dt.groupby("slot").r15.rank(ascending=False, method="first")
fp = dt[(dt["rank"] <= 5) & (dt.ev == 0)]
print("DT pairs", dt.slot.nunique(), " evidence hands with dir==tdir:", round((dt[dt.ev == 1].dir == dt[dt.ev == 1].tdir).mean(), 3), " dir==0:", round((dt[dt.ev == 1].dir == 0).mean(), 3))
print("r15 top-5 FPs:", len(fp), " reverse-direction share:", round((fp.dir == -fp.tdir).mean(), 3), " no-flow share:", round((fp.dir == 0).mean(), 3), " same-direction share:", round((fp.dir == fp.tdir).mean(), 3))
def ap5(g, col):
    top = g.sort_values(col, ascending=False, kind="mergesort").ev.values[:5]; hits = 0; s = 0.0
    for i, e in enumerate(top):
        if e: hits += 1; s += hits / (i + 1)
    return s / min(5, int(g.m_p.iloc[0]))
E = lambda df, c: df.groupby("slot").apply(lambda g: ap5(g, c)).mean()
dt["orc"] = dt.r15 - 10 * (dt.dir != dt.tdir)
print(f"DT E r15 {E(dt, 'r15'):.4f}  | oracle demote non-true-direction {E(dt, 'orc'):.4f}")
# label-free direction inference: sign of summed flow over the pair's candidate hands weighted by r15 score
for nm, est in [("sum flow over top-10 candidates", dt[dt["rank"] <= 10].groupby("slot").dir.sum().apply(np.sign)),
                ("o_dir_agree mean sign", dt.groupby("slot").o_dir_agree.mean().apply(np.sign))]:
    acc = (est.reindex(truth.index) == truth).mean(); print(f"direction estimate [{nm}] accuracy {acc:.3f}")
    dt["est"] = dt.slot.map(est); dt["dem"] = dt.r15 - 0.3 * (dt.dir == -dt.est)
    print(f"   demote reverse-to-estimate by 0.3: DT E {E(dt, 'dem'):.4f}")
