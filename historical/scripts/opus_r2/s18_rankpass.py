"""Is evidence_rank = (planting pass/type, time)?  Pairwise rank-time concordance within vs across candidate 'type' partitions."""
import numpy as np, pandas as pd, itertools
RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
ev = pd.read_csv(f"{RAW}/development_evidence.csv"); lab = pd.read_csv(f"{RAW}/development_labels.csv")
ev = ev.merge(lab[["pair_id", "player_1", "player_2"]], on="pair_id")
hids = ev.hand_id.unique().tolist()
H = pd.read_parquet(f"{RAW}/hands.parquet", columns=["hand_id", "started_at", "final_pot", "big_blind", "players_at_showdown", "board_cards"], filters=[("hand_id", "in", hids)])
S = pd.read_parquet(f"{RAW}/seats.parquet", filters=[("hand_id", "in", hids)])
Ac = pd.read_parquet(f"{RAW}/actions.parquet", filters=[("hand_id", "in", hids)])
ev = ev.merge(H, on="hand_id"); ev["t"] = ev.started_at.astype("int64")
S = S.set_index(["hand_id", "player_id"])
def typ(r):
    a = Ac[Ac.hand_id == r.hand_id].sort_values("action_no")
    p1, p2 = r.player_1, r.player_2
    s1 = S.loc[(r.hand_id, p1)]; s2 = S.loc[(r.hand_id, p2)]
    last_street = a.street.iloc[-1]
    # who folded among the pair and on which street
    f1 = a[(a.player_id == p1) & (a.action == "fold")]; f2 = a[(a.player_id == p2) & (a.action == "fold")]
    fold_st = (f1.street.iloc[0] if len(f1) else (f2.street.iloc[0] if len(f2) else "none"))
    pair_aggr = a[a.player_id.isin([p1, p2]) & a.action.isin(["bet", "raise", "all_in"])]
    return pd.Series(dict(sd=int(r.players_at_showdown > 0), both_sd=int(s1.went_to_showdown and s2.went_to_showdown),
                          any_fold=int(s1.folded or s2.folded), fold_street=fold_st, end_street=last_street,
                          n_pair_aggr=min(len(pair_aggr), 3), preflop_only=int(a.street.nunique() == 1)))
T = ev.apply(typ, axis=1); ev = pd.concat([ev, T], axis=1)
def concord(g, key=None):
    c = t = 0
    for (i, a), (j, b) in itertools.combinations(g.iterrows(), 2):
        if key is not None and a[key] != b[key]: continue
        t += 1; c += int((a.evidence_rank < b.evidence_rank) == (a.t < b.t))
    return c, t
for fm, G in ev.groupby("behavior_family"):
    out = {}
    for key in [None, "sd", "both_sd", "any_fold", "fold_street", "end_street", "n_pair_aggr", "preflop_only"]:
        c = t = 0
        for _, g in G.groupby("pair_id"):
            a, b = concord(g, key); c += a; t += b
        out[str(key)] = (round(c / max(t, 1), 3), t)
    print(fm[:2], out, flush=True)
    # across-type ordering: does type A come before type B in rank regardless of time?
    for key in ["sd", "fold_street", "end_street"]:
        print("   mean rank by", key, G.groupby(key).evidence_rank.agg(["mean", "size"]).round(2).to_dict("index"))
