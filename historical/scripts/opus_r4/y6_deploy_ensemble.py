"""R4-Y6: deploy an ENSEMBLE of event-model feature variants to eval and write an evidence patch.
Variant = (dev new-block parquet, dev role parquet, eval new-block parquet, eval role parquet); every variant is trained on the dev right-censored rows (3 seeds) and predicts all
co-seated eval hands; event probabilities are averaged over variants and seeds, then first-five DP, then either the rank average with the frozen R15 top-20 (MODE=rank, PAR=w)
or logit(TabICL eval probability) + PAR * logit(q) (MODE=stack). No submission side effects.
Usage: python y6_deploy_ensemble.py <family> <rank|stack> <PAR> <eval_full.parquet> <eval_candidates.parquet> <out_patch.csv> <variant> [<variant> ...]
       variant = devnew:devrole:evalnew:evalrole (paths)"""
import numpy as np, pandas as pd, lightgbm as lgb, json, sys, hashlib
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18")
import ci_censored_event as C
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
fam, MODE, PAR = sys.argv[1], sys.argv[2], float(sys.argv[3]); f_full, f_cand, out_csv = sys.argv[4:7]; variants = [v.split(":") for v in sys.argv[7:]]
def add_cells(df):
    cells = {"sp": df.both_flop.astype(float), "ci": ((df.pa_at_trig == 6) & df.y1.isin([2, 3])).astype(float), "fold": df.x_s_fold_to_r.astype(float),
             "sfold": (df.x_s_fold_to_r * (df.x_hsS_last >= 0.55)).astype(float), "hu": (df.both_flop & df.all_out_folded).astype(float)}
    out = []
    for nm, v in cells.items():
        df[f"c_{nm}"] = v; df[f"c_k_{nm}"] = v.groupby(df.slot).cumsum() - v; df[f"c_n_{nm}"] = v.groupby(df.slot).transform("sum")
        df[f"c_rel_{nm}"] = (df[f"c_k_{nm}"] + 0.5) / df[f"c_n_{nm}"].clip(lower=1); out += [f"c_{nm}", f"c_k_{nm}", f"c_n_{nm}", f"c_rel_{nm}"]
    return out
def assemble(frame, newf, role):
    NEW = [c for c in newf.columns if c not in ("slot", "h", "pa", "pb", "pair_id")]; ROLE = [c for c in role.columns if c.startswith("x_") and c not in ("x_k", "x_n")]
    d = frame.merge(newf[["slot", "h"] + NEW], on=["slot", "h"], how="left", validate="one_to_one"); d[NEW] = d[NEW].fillna(0.0)
    d = d.merge(role[["slot", "h"] + ROLE], on=["slot", "h"], how="left", validate="one_to_one"); assert d[ROLE].notna().all().all(), "role features missing"
    d = d.sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True); CE = add_cells(d); return d, C.FEATURES + NEW + ROLE + CE
dev_all = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet")); dev_all = dev_all[dev_all.fam == fam].reset_index(drop=True); ev_all = C.prepare(pd.read_parquet(f_full))
p = None; ev0 = None; info = []
for dn, dr, en, er in variants:
    dev, FS = assemble(dev_all.copy(), pd.read_parquet(dn), pd.read_parquet(dr)); ev, FS2 = assemble(ev_all.copy(), pd.read_parquet(en), pd.read_parquet(er)); assert FS == FS2, "dev/eval feature lists differ"
    if ev0 is None: ev0 = ev
    assert (ev.slot.values == ev0.slot.values).all() and (ev.h.values == ev0.h.values).all(), "variant row order differs"
    inc = C.uncensored_training_rows(dev); pv = np.zeros(len(ev))
    for seed in (27, 28, 29):
        m = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": 4, "random_state": seed}); m.fit(dev.loc[inc, FS].astype(float), dev.loc[inc, "ev"]); pv += m.predict_proba(ev[FS].astype(float))[:, 1] / 3
    info.append(dict(dev_new=dn.split("/")[-1], dev_role=dr.split("/")[-1], features=len(FS), eval_p_mean=float(pv.mean()))); p = pv if p is None else p + pv
    print(f"variant {info[-1]}", flush=True)
p /= len(variants); ev = ev0; cand = pd.read_parquet(f_cand); j = C.rank_candidates(ev, cand, C.first_k_marginal(ev, p), weight=PAR if MODE == "rank" else 0.5)
if MODE == "stack":
    tab = pd.read_csv("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round15_campaign/tabicl_eval/predictions.csv.gz").rename(columns={"score": "tab"})
    j = j.merge(tab[["slot", "hand_id", "tab"]], on=["slot", "hand_id"], how="left", validate="one_to_one"); assert j.tab.notna().all()
    lg = lambda v: np.log(np.clip(v, 1e-5, 1 - 1e-5) / (1 - np.clip(v, 1e-5, 1 - 1e-5))); j["newscore"] = lg(j.tab.values) + PAR * lg(j.q.values)
z = j.sort_values(["slot", "newscore", "ts", "h"], ascending=[True, False, True, True], kind="stable").groupby("slot", sort=False).head(5).copy()
z["rank"] = z.groupby("pair_id").cumcount() + 1; pt = z.pivot(index="pair_id", columns="rank", values="hand_id"); pt.columns = [f"evidence_hand_{i}" for i in pt.columns]; out = pt.reset_index()
assert out.notna().all().all() and len(out) == cand.pair_id.nunique(); out.to_csv(out_csv, index=False)
old5 = cand[cand.r <= 5].groupby("pair_id").hand_id.apply(set); ov = [len(set(out.set_index("pair_id").loc[p_].values) & old5[p_]) for p_ in old5.index]
rec = dict(family=fam, mode=MODE, par=PAR, variants=info, pairs=len(out), mean_overlap_with_R15_top5=float(np.mean(ov)), pairs_changed_vs_R15=int(np.sum(np.array(ov) < 5)), sha256=hashlib.sha256(open(out_csv, "rb").read()).hexdigest())
json.dump(rec, open(out_csv.replace(".csv", ".receipt.json"), "w"), indent=1); print(json.dumps(rec))
