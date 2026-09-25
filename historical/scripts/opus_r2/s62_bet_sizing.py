"""Bet sizing as unused information: distribution of aggressive-action size ratios (added chips / pot_before and
raise-to / to_call) by pair members in evidence hands vs non-evidence candidate hands vs population; look for spikes."""
import numpy as np, pandas as pd
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; RAW = f"{A_}/data/raw"
lab = pd.read_csv(f"{RAW}/development_labels.csv"); ev = pd.read_csv(f"{RAW}/development_evidence.csv")
d = pd.read_parquet(f"{A_}/round11_scoped/dev_oof_aligned.parquet")[["slot", "hand_id", "ev"]]
cand = pd.read_csv(f"{A_}/round3_research_20260917/r6_narrow_candidates_v2.csv", usecols=["slot", "hand_id", "pair_player_lo", "pair_player_hi", "family"])
d = d.merge(cand, on=["slot", "hand_id"])
hids = d.hand_id.unique().tolist()
Ac = pd.read_parquet(f"{RAW}/actions.parquet", filters=[("hand_id", "in", hids)])
Ac = Ac[Ac.action.isin(["bet", "raise", "all_in"])].copy()
Ac["ratio_pot"] = Ac.amount / Ac.pot_before.clip(lower=1); Ac["raise_mult"] = Ac.amount_to / Ac.to_call.clip(lower=1)
m = Ac.merge(d, on="hand_id")
m = m[(m.player_id == m.pair_player_lo) | (m.player_id == m.pair_player_hi)]
print("pair-member aggressive actions in candidate hands:", len(m), " in evidence hands:", int(m.ev.sum()))
for fam, g in m.groupby("family"):
    a = g[g.ev == 1].ratio_pot.round(2); b = g[g.ev == 0].ratio_pot.round(2)
    va = a.value_counts(normalize=True).head(8); vb = b.value_counts(normalize=True)
    print(f"== {fam[:2]}: top sizing ratios (added/pot) in EVIDENCE hands vs share among non-evidence candidates")
    print("   ", {k: (round(v, 3), round(vb.get(k, 0), 3)) for k, v in va.items()})
    print(f"    median ratio ev {a.median():.2f} non-ev {b.median():.2f};  bets on preflop share ev {(g[g.ev == 1].street == 'preflop').mean():.2f} non-ev {(g[g.ev == 0].street == 'preflop').mean():.2f}")
# population reference (random 200k aggressive actions)
pop = pd.read_parquet(f"{RAW}/actions.parquet", columns=["action", "amount", "pot_before"]).sample(400000, random_state=0)
pop = pop[pop.action.isin(["bet", "raise", "all_in"])]; pr = (pop.amount / pop.pot_before.clip(lower=1)).round(2).value_counts(normalize=True).head(10)
print("population top sizing ratios:", pr.round(3).to_dict())
