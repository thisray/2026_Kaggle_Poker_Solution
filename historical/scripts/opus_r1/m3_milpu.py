import numpy as np, pandas as pd, time, lightgbm as lgb, os, gc
from sklearn.metrics import average_precision_score, roc_auc_score
import handfeat as HF, pairindex as PI
OUT = HF.OUT; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet")   # key,pool,p_lo,p_hi,n,label,fam,touch_pos,fold,oof  (174000 rows, slot order)
ev = pd.read_parquet(f"{OUT}/m1_eval_scores.parquet")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
for d in (dv, ev):
    d["slot"] = PI.pair_slot(d.pool.values, loc.local.loc[d.p_lo].values, loc.local.loc[d.p_hi].values)
dv["pop"] = (dv.n >= 57) & ((~dv.touch_pos) | (dv.label >= 0))
fold_of_pool = dv.groupby("pool").fold.first().reindex(range(400)).values
hidx = pd.read_parquet(f"{OUT}/np/hand_index.parquet"); hmap = dict(zip(hidx.hand_id, hidx.hi))
labels = pd.read_csv(f"{RAW}/development_labels.csv"); evid = pd.read_csv(f"{RAW}/development_evidence.csv")
pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
labels["key"] = np.minimum(labels.player_1.map(pmap), labels.player_2.map(pmap)) * 12000 + np.maximum(labels.player_1.map(pmap), labels.player_2.map(pmap))
key2slot = dict(zip(dv.key, dv.slot)); pid2slot = dict(zip(labels.pair_id, labels.key.map(key2slot)))
ev_set = set(zip(evid.pair_id.map(pid2slot), evid.hand_id.map(hmap)))
slot_label = dv.set_index("slot").label; slot_fam = dv.set_index("slot").fam; slot_fold = dv.set_index("slot").fold; slot_oof = dv.set_index("slot").oof
# ---------- enumerate pair-hands
Hd, Sd, Td, SLd = PI.all_pair_hands(0); log("dev pair-hands", len(Hd))
He, Se, Te, SLe = PI.all_pair_hands(1); log("eval pair-hands", len(He))
# ---------- training rows
lab_d = slot_label.reindex(SLd).values; lab_e = slot_label.reindex(SLe).values
rng = np.random.RandomState(0)
u_pairs = dv[dv["pop"] & (dv.label == -1) & (dv.oof < 0.02)].slot.values
u_pairs = set(rng.choice(u_pairs, 8000, replace=False))
m_pos = lab_d == 1; m_neg = lab_d == 0; m_u = np.isin(SLd, list(u_pairs))
m_e_lab = (lab_e >= 0)   # labelled pairs' eval-phase hands: non-colluding phase -> negatives
idx_d = np.where(m_pos | m_neg | m_u)[0]; idx_e = np.where(m_e_lab)[0]
h_tr = np.r_[Hd[idx_d], He[idx_e]]; s_tr = np.r_[Sd[idx_d], Se[idx_e]]; t_tr = np.r_[Td[idx_d], Te[idx_e]]; sl_tr = np.r_[SLd[idx_d], SLe[idx_e]]
ph_tr = np.r_[np.zeros(len(idx_d), np.int8), np.ones(len(idx_e), np.int8)]
X = HF.features(h_tr, s_tr, t_tr); log("train features", X.shape)
ybag = np.r_[(lab_d[idx_d] == 1).astype(int), np.zeros(len(idx_e), int)]
is_ev = np.array([(sl, hh) in ev_set for sl, hh in zip(sl_tr, h_tr)])
fold = fold_of_pool[sl_tr // 900]
fam = slot_fam.reindex(sl_tr).values
src = np.where(ph_tr == 1, "eval_lab", np.where(lab_d[idx_d] == 1, "pos", np.where(lab_d[idx_d] == 0, "neg", "U")) if False else None)
src = np.empty(len(h_tr), dtype=object)
src[:len(idx_d)] = np.where(lab_d[idx_d] == 1, "pos", np.where(lab_d[idx_d] == 0, "neg", "U")); src[len(idx_d):] = "eval_lab"
log("rows: pos-bag", ybag.sum(), "ev", is_ev.sum(), "neg", (ybag == 0).sum(), pd.Series(src).value_counts().to_dict())
params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1, lambda_l2=2.0, verbose=-1, num_threads=18, seed=11)
def weights(y):
    w = np.ones(len(y)); npos = y.sum(); nneg = len(y) - npos
    w[y == 0] = npos / nneg
    return w
def oof_train(y, mask, rounds=400, tag=""):
    oof = np.zeros(len(y)); models = []
    for f in range(5):
        tr = mask & (fold != f)
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr], weight=weights(y[tr])), num_boost_round=rounds)
        va = fold == f
        oof[va] = m.predict(X[va]); models.append(m)
    log(tag, "trained")
    return oof, models
def report(score, tag):
    negm = ybag == 0
    m = is_ev | negm
    log(f"{tag}: evidence-vs-neg AUC {roc_auc_score(is_ev[m], score[m]):.4f} AP {average_precision_score(is_ev[m], score[m]):.4f}")
    for fm in ["directed_transfer","soft_play","coordinated_isolation"]:
        mm = (is_ev & (fam == fm)) | negm
        print(f"   {fm} ev AP {average_precision_score(is_ev[mm], score[mm]):.4f}", flush=True)
    # MAP@5 on positive pairs
    df = pd.DataFrame({"sl": sl_tr, "h": h_tr, "s": score, "ev": is_ev, "fam": fam})[ybag == 1]
    aps = []
    for sl, g in df.groupby("sl"):
        rel = set(g.h[g.ev]); top = g.sort_values("s", ascending=False).h.values[:5]
        hits = 0; ssum = 0.0
        for i, hh in enumerate(top):
            if hh in rel: hits += 1; ssum += hits / (i + 1)
        aps.append((g.fam.iloc[0], ssum / min(5, len(rel))))
    a = pd.DataFrame(aps, columns=["fam", "ap"])
    print(f"   MAP@5 {a.ap.mean():.4f}", a.groupby("fam").ap.mean().round(4).to_dict(), flush=True)
    # expected planted count per pos pair: hands in pos bags above neg 99.9% quantile
    q = np.quantile(score[negm], [0.999, 0.9999])
    df2 = pd.DataFrame({"sl": sl_tr, "s": score})[ybag == 1]
    print(f"   neg q999={q[0]:.4f} q9999={q[1]:.4f}; pos-bag hands above q999 per pair: {np.mean(df2.groupby('sl').s.apply(lambda x: (x > q[0]).sum())):.2f}; above q9999: {np.mean(df2.groupby('sl').s.apply(lambda x: (x > q[1]).sum())):.2f}", flush=True)
# round 0: evidence-only positives (baseline)
y0 = is_ev.astype(int); mask0 = is_ev | (ybag == 0)
oof0, _ = oof_train(y0, mask0, tag="r0 evidence-only")
report(oof0, "r0 evidence-only")
# round 1: bag labels
oof1, _ = oof_train(ybag, np.ones(len(ybag), bool), tag="r1 bag")
report(oof1, "r1 bag")
# round 2: self-training: pseudo positives = evidence + pos-bag hands above neg q99.9 of r1; drop other pos-bag hands
thr = np.quantile(oof1[ybag == 0], 0.999)
pseudo = is_ev | ((ybag == 1) & (oof1 > thr))
mask2 = pseudo | (ybag == 0)
log("r2 pseudo positives", pseudo.sum())
oof2, models2 = oof_train(pseudo.astype(int), mask2, tag="r2 self-train")
report(oof2, "r2 self-train")
thr2 = np.quantile(oof2[ybag == 0], 0.999)
pseudo3 = is_ev | ((ybag == 1) & (oof2 > thr2))
mask3 = pseudo3 | (ybag == 0)
log("r3 pseudo positives", pseudo3.sum())
oof3, models3 = oof_train(pseudo3.astype(int), mask3, tag="r3 self-train")
report(oof3, "r3 self-train")
np.save(f"{OUT}/m3_train_oof.npy", np.c_[oof0, oof1, oof2, oof3])
pd.DataFrame({"slot": sl_tr, "h": h_tr, "phase": ph_tr, "ybag": ybag, "is_ev": is_ev, "fold": fold, "src": src}).to_parquet(f"{OUT}/m3_train_meta.parquet")
for f, m in enumerate(models3): m.save_model(f"{OUT}/m3_hand_r3_f{f}.txt")
imp = pd.Series(sum(m.feature_importance("gain") for m in models3), index=X.columns).sort_values(ascending=False)
print(imp.head(25).round(0).to_string())
