"""What orders evidence_rank?  If not time, it may be a severity measure that also defines 'qualifying'."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
ev = pd.read_csv(f"{RAW}/development_evidence.csv"); lab = pd.read_csv(f"{RAW}/development_labels.csv")
ev = ev.merge(lab[["pair_id", "player_1", "player_2"]], on="pair_id")
H = pd.read_parquet(f"{RAW}/hands.parquet", columns=["hand_id", "started_at", "final_pot", "big_blind", "players_at_showdown"])
S = pd.read_parquet(f"{RAW}/seats.parquet", columns=["hand_id", "player_id", "net_chips", "total_contribution", "went_to_showdown", "folded"])
ev = ev.merge(H, on="hand_id")
s1 = S.rename(columns={c: c + "_1" for c in ["player_id", "net_chips", "total_contribution", "went_to_showdown", "folded"]})
s2 = S.rename(columns={c: c + "_2" for c in ["player_id", "net_chips", "total_contribution", "went_to_showdown", "folded"]})
ev = ev.merge(s1, left_on=["hand_id", "player_1"], right_on=["hand_id", "player_id_1"]).merge(s2, left_on=["hand_id", "player_2"], right_on=["hand_id", "player_id_2"])
ev["pot_bb"] = ev.final_pot / ev.big_blind
ev["pair_net_abs_bb"] = (ev.net_chips_1 - ev.net_chips_2).abs() / ev.big_blind / 2
ev["pair_sum_net_bb"] = (ev.net_chips_1 + ev.net_chips_2) / ev.big_blind
ev["contrib_min_bb"] = np.minimum(ev.total_contribution_1, ev.total_contribution_2) / ev.big_blind
ev["contrib_max_bb"] = np.maximum(ev.total_contribution_1, ev.total_contribution_2) / ev.big_blind
ev["t"] = ev.started_at.astype("int64")
ev["trank"] = ev.groupby("pair_id").t.rank()
for fm, g in ev.groupby("behavior_family"):
    print(f"== {fm}: n {len(g)} pairs {g.pair_id.nunique()}")
    for c in ["trank", "pot_bb", "final_pot", "pair_net_abs_bb", "pair_sum_net_bb", "contrib_min_bb", "contrib_max_bb", "players_at_showdown"]:
        # within-pair spearman averaged
        rs = [spearmanr(x.evidence_rank, x[c]).correlation for _, x in g.groupby("pair_id") if len(x) >= 3 and x[c].nunique() > 1]
        print(f"   rank vs {c:20s} mean within-pair spearman {np.nanmean(rs):+.3f}  (n pairs {len(rs)})")
    print("   rank==time-order share:", round((g.evidence_rank == g.trank).mean(), 3))
print("\n==== composite sort keys: share of pairs whose evidence_rank order is exactly reproduced")
ev["sd"] = (ev.players_at_showdown > 0).astype(int)
ev["pair_sd"] = (ev.went_to_showdown_1 & ev.went_to_showdown_2).astype(int)
ev["any_fold"] = (ev.folded_1 | ev.folded_2).astype(int)
ev["both_fold"] = (ev.folded_1 & ev.folded_2).astype(int)
ev["pot_q"] = ev.pot_bb
keys = {"time": ["t"], "sd,time": ["sd", "t"], "pair_sd,time": ["pair_sd", "t"], "any_fold desc,time": ["nf", "t"], "pot": ["pot_q"], "sd,pot": ["sd", "pot_q"]}
ev["nf"] = -ev.any_fold
for fm, g in ev.groupby("behavior_family"):
    out = {}
    for nm, k in keys.items():
        ok = []
        for _, x in g.groupby("pair_id"):
            if len(x) < 2: continue
            ok.append((x.sort_values(k, kind="mergesort").evidence_rank.values == np.sort(x.evidence_rank.values)).all())
        out[nm] = round(np.mean(ok), 3)
    print(fm[:2], out)
    print("   rank x sd crosstab:\n", pd.crosstab(g.evidence_rank, g.sd).to_string())
    print("   rank x any_fold crosstab:\n", pd.crosstab(g.evidence_rank, g.any_fold).to_string())
