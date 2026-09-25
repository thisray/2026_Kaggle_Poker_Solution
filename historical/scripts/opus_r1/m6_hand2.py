"""Hand-level evidence detector with surprisal features; OOF by pool; MAP@5 and first-k selection analysis."""
import numpy as np, pandas as pd, time, lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
import handfeat as HF, handfeat2 as HF2, pairindex as PI
OUT = HF.OUT; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
dv["slot"] = PI.pair_slot(dv.pool.values, loc.local.loc[dv.p_lo].values, loc.local.loc[dv.p_hi].values)
fold_of_pool = dv.groupby("pool").fold.first().reindex(range(400)).values
hidx = pd.read_parquet(f"{OUT}/np/hand_index.parquet"); hmap = dict(zip(hidx.hand_id, hidx.hi))
labels = pd.read_csv(f"{RAW}/development_labels.csv"); evid = pd.read_csv(f"{RAW}/development_evidence.csv")
pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
labels["key"] = np.minimum(labels.player_1.map(pmap), labels.player_2.map(pmap)) * 12000 + np.maximum(labels.player_1.map(pmap), labels.player_2.map(pmap))
key2slot = dict(zip(dv.key, dv.slot)); pid2slot = dict(zip(labels.pair_id, labels.key.map(key2slot)))
evid["slot"] = evid.pair_id.map(pid2slot); evid["h"] = evid.hand_id.map(hmap)
ev_rank = dict(zip(zip(evid.slot, evid.h), evid.evidence_rank))
slot_label = dv.set_index("slot").label; slot_fam = dv.set_index("slot").fam
Hd, Sd, Td, SLd = PI.all_pair_hands(0); He, Se, Te, SLe = PI.all_pair_hands(1)
lab_d = slot_label.reindex(SLd).values; lab_e = slot_label.reindex(SLe).values
rng = np.random.RandomState(0)
u_pairs = dv[(dv.n >= 57) & (~dv.touch_pos) & (dv.label == -1) & (dv.oof < 0.02)].slot.values
u_pairs = rng.choice(u_pairs, 6000, replace=False)
m_u = np.isin(SLd, u_pairs)
idx_d = np.where((lab_d >= 0) | m_u)[0]; idx_e = np.where(lab_e >= 0)[0]
h = np.r_[Hd[idx_d], He[idx_e]]; s = np.r_[Sd[idx_d], Se[idx_e]]; t = np.r_[Td[idx_d], Te[idx_e]]; sl = np.r_[SLd[idx_d], SLe[idx_e]]
phase = np.r_[np.zeros(len(idx_d), np.int8), np.ones(len(idx_e), np.int8)]
X = HF2.features(h, s, t); log("features", X.shape)
lab = np.r_[lab_d[idx_d], lab_e[idx_e]]
rk = np.array([ev_rank.get((a, b), 0) for a, b in zip(sl, h)])
is_ev = rk > 0
pos_bag = (lab == 1) & (phase == 0)
neg = ~pos_bag
fold = fold_of_pool[sl // 900]; fam = slot_fam.reindex(sl).values
params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=40, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, num_threads=18, seed=5)
y = is_ev.astype(int); trainable = is_ev | neg
oof = np.zeros(len(y))
for f in range(5):
    tr = trainable & (fold != f)
    w = np.where(y[tr] == 1, 1.0, y[tr].sum() / (len(y[tr]) - y[tr].sum()) * 3)
    mdl = lgb.train(params, lgb.Dataset(X[tr], y[tr], weight=w), num_boost_round=600)
    va = fold == f; oof[va] = mdl.predict(X[va])
log("trained")
m = trainable
log(f"hand ev-vs-neg AUC {roc_auc_score(y[m], oof[m]):.4f} AP {average_precision_score(y[m], oof[m]):.4f}")
for fm in ["directed_transfer","soft_play","coordinated_isolation"]:
    mm = (is_ev & (fam == fm)) | neg
    print(f"   {fm} AP {average_precision_score(y[mm], oof[mm]):.4f}", flush=True)
D = pd.DataFrame({"sl": sl, "h": h, "s": oof, "ev": is_ev, "rk": rk, "fam": fam, "pos": pos_bag, "phase": phase})
ts = np.load(f"{OUT}/np/h_ts.npy"); D["ts"] = ts[D.h.values]
P = D[D.pos].copy()
def map5(df, col):
    res = []
    for k, g in df.groupby("sl"):
        rel = set(g.h[g.ev]); top = g.sort_values(col, ascending=False).h.values[:5]
        hits = 0; ssum = 0.0
        for i, hh in enumerate(top):
            if hh in rel: hits += 1; ssum += hits / (i + 1)
        res.append((g.fam.iloc[0], ssum / min(5, len(rel))))
    r = pd.DataFrame(res, columns=["fam", "ap"]); return r.ap.mean(), r.groupby("fam").ap.mean().round(4).to_dict()
print("MAP@5 score only:", map5(P, "s"), flush=True)
# oracle: best possible MAP@5 if we knew planted set = top-q hands; and time-prior variants
P["tpct"] = P.groupby("sl").ts.rank(pct=True)
P = P.sort_values(["sl", "ts"])
# expected number of earlier candidates using calibrated-ish probability: p = s (clipped)
P["cum_before"] = P.groupby("sl").s.cumsum() - P.s
from scipy.stats import poisson
for lam_scale in [0.5, 1.0, 2.0]:
    P["prior"] = poisson.cdf(4, P.cum_before * lam_scale)
    P["s_first5"] = P.s * P.prior
    print(f"MAP@5 score x P(<5 earlier candidates) scale={lam_scale}:", map5(P, "s_first5"), flush=True)
for a in [0.5, 1.0, 2.0]:
    P["s_time"] = P.s * np.exp(-a * P.tpct)
    print(f"MAP@5 score x exp(-{a} tpct):", map5(P, "s_time"), flush=True)
# how well do high-score hands cover evidence: recall of evidence within top-K by score
for K in [5, 10, 20, 40]:
    rec = P.groupby("sl").apply(lambda g: g.sort_values("s", ascending=False).head(K).ev.sum() / max(g.ev.sum(), 1)).mean()
    print(f"evidence recall@{K} by score: {rec:.3f}")
q999 = np.quantile(D.s[D.phase.eq(0) & ~D.pos], 0.999)
P["hi"] = P.s > q999
print("per pos pair: hands above neg q999:", P.groupby("sl").hi.sum().describe().round(2).to_dict())
print("evidence hands above q999 frac:", P.hi[P.ev].mean())
D.to_parquet(f"{OUT}/m6_handscores.parquet")
