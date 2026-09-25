"""Colluder-type persistence across phases: are eval pairs that involve a player who colluded in DEV (hidden, unlabelled,
dev pair OOF high) more often eval positives (proxy: eval top-ranked by the independent eval-phase P model)?"""
import numpy as np, pandas as pd
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; RAW = f"{A_}/data/raw"
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet")          # dev-phase pair OOF (p_lo, p_hi are player indices)
lab = pd.read_csv(f"{RAW}/development_labels.csv"); pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
pos_players = set(lab[lab.label == 1].player_1.map(pmap)) | set(lab[lab.label == 1].player_2.map(pmap))
hid = dv[(dv.label == -1) & (dv.oof > 0.5)]
hid_players = set(hid.p_lo) | set(hid.p_hi)
print("dev hidden-like pairs (unlabelled, oof>0.5):", len(hid), " players:", len(hid_players), " overlap with labelled-positive players:", len(hid_players & pos_players))
ev = pd.read_csv(f"{RAW}/evaluation_pairs.csv"); ev["a"] = ev.player_1.map(pmap); ev["b"] = ev.player_2.map(pmap)
r = pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv")[["pair_id", "risk_score"]]; ev = ev.merge(r, on="pair_id")
ev = ev.sort_values(["risk_score", "pair_id"], ascending=[False, True]).reset_index(drop=True); ev["rk"] = np.arange(1, len(ev) + 1)
ev["touch_hidden"] = ev.a.isin(hid_players) | ev.b.isin(hid_players)
ev["touch_labpos"] = ev.a.isin(pos_players) | ev.b.isin(pos_players)
print("eval pairs touching a labelled-positive player:", int(ev.touch_labpos.sum()), "(should be 0 by construction)")
g = ev.groupby("touch_hidden").agg(n=("rk", "size"), top450_rate=("rk", lambda s: (s <= 450).mean()), top1000_rate=("rk", lambda s: (s <= 1000).mean()), median_rank=("rk", "median"))
print(g)
# also: any dev-phase high-oof player (labelled or not) -- but labelled ones are excluded; use dev oof>0.2 unlabelled
for thr in [0.2, 0.05]:
    hp = dv[(dv.label == -1) & (dv.oof > thr)]; pl = set(hp.p_lo) | set(hp.p_hi)
    t = ev.a.isin(pl) | ev.b.isin(pl)
    print(f"dev unlabelled oof>{thr} players: {len(pl)}; eval pairs touching: {int(t.sum())}; top450 rate touching {(ev[t].rk <= 450).mean():.4f} vs not {(ev[~t].rk <= 450).mean():.4f}")
