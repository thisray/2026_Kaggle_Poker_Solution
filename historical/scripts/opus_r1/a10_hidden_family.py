"""Are dev hidden positives (unlabelled, high OOF) a different family? Adversarial validation vs labelled positives on the same dev table."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from pairfeat import build
from pairfeat2 import build2
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
tr = pd.read_parquet(f"{OUT}/m5_both_train_oof.parquet")
res = {}
FAMS = ["directed_transfer", "soft_play", "coordinated_isolation"]
for nm in ["devsub11", "devsub12"]:
    t = pd.read_parquet(f"{OUT}/ptab_{nm}.parquet"); t["key"] = t.p_lo * 12000 + t.p_hi
    r = tr[tr.src == nm]
    labk = r[r.y == 1][["key", "fam"]]; hidk = r[(r.label == -1) & (r.oof > 0.5)][["key"]]
    A = t.set_index("key").loc[labk.key].reset_index(); B = t.set_index("key").loc[hidk.key].reset_index()
    XA = pd.concat([build(A), build2(A)], axis=1); XB = pd.concat([build(B), build2(B)], axis=1)
    X = pd.concat([XA, XB], ignore_index=True).drop(columns=["n", "lo_hands", "hi_hands"]); y = np.r_[np.zeros(len(XA)), np.ones(len(XB))]
    oof = np.zeros(len(y))
    for trn, val in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=10, feature_fraction=0.7, verbose=-1, num_threads=4), lgb.Dataset(X.iloc[trn], y[trn]), 200)
        oof[val] = m.predict(X.iloc[val])
    print(nm, "labelled pos", len(XA), "hidden", len(XB), "adversarial AUC:", round(roc_auc_score(y, oof), 4))
    imp = pd.Series(m.feature_importance("gain"), index=X.columns).sort_values(ascending=False).head(10); print(imp.round(0).to_string())
    # family classifier trained on labelled, applied to hidden
    yf = labk.fam.map({f: i for i, f in enumerate(FAMS)}).values
    fm = lgb.train(dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=15, min_data_in_leaf=10, feature_fraction=0.6, verbose=-1, num_threads=4), lgb.Dataset(XA.drop(columns=["n", "lo_hands", "hi_hands"]), yf), 300)
    ph = fm.predict(XB.drop(columns=["n", "lo_hands", "hi_hands"]))
    print("   hidden predicted family:", pd.Series(np.array(FAMS)[ph.argmax(1)]).value_counts().to_dict(), " pmax quantiles:", np.round(np.quantile(ph.max(1), [0.1, 0.25, 0.5]), 3))
    # compare key rates between labelled CI, labelled all, hidden
    for c in ["iso_ofold_sum_rate", "squeeze_cr_sym", "fold_to_strong_hs_z_max", "check_hu_strong_eq_z_max", "n_hi5_act_z_max", "weak_aggr_act_z_max", "call_to_dead_eq_z_max", "xs_pass_hu_c_max", "both_vpip_rate", "hu_noaggr_cr_sym"]:
        if c in XA.columns:
            print(f"   {c:26s} lab-DT {XA[c][yf==0].mean():8.4f} lab-SP {XA[c][yf==1].mean():8.4f} lab-CI {XA[c][yf==2].mean():8.4f} hidden {XB[c].mean():8.4f}")
    B.assign(adv=oof[len(XA):], fam_pred=np.array(FAMS)[ph.argmax(1)], pmax=ph.max(1)).to_parquet(f"{OUT}/a10_hidden_{nm}.parquet")
