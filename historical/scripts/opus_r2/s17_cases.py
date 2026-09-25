"""Print raw hands: low-score evidence (missed) vs high-score in-window non-evidence (false alarms), DT and SP."""
import numpy as np, pandas as pd
A = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A}/opus_r1_20260917"; RAW = f"{A}/data/raw"
M = pd.read_parquet(f"{OUT}/s6_situation.parquet")
hidx = pd.read_parquet(f"{OUT}/np/hand_index.parquet"); hi2id = dict(zip(hidx.hi, hidx.hand_id))
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet"); gi2pid = dict(zip(pidx.pi, pidx.player_id))
mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
M["hand_id"] = M.h.map(hi2id); M["pa"] = [gi2pid[x] for x in mem[M.sl // 900, (M.sl % 900) // 30]]; M["pb"] = [gi2pid[x] for x in mem[M.sl // 900, M.sl % 30]]
rng = np.random.default_rng(1)
pick = []
for fm in ["directed_transfer", "soft_play"]:
    F = M[(M.fam == fm) & M.win & M.sit]
    a = F[F.ev & (F.sc_fam < 0.05)]; b = F[~F.ev & (F.sc_fam > 0.85)]
    pick += [("MISSED-EV " + fm, r) for r in a.sample(min(4, len(a)), random_state=1).itertuples()]
    pick += [("FALSE-ALARM " + fm, r) for r in b.sample(min(3, len(b)), random_state=1).itertuples()]
hids = [r.hand_id for _, r in pick]
H = pd.read_parquet(f"{RAW}/hands.parquet", filters=[("hand_id", "in", hids)])
S = pd.read_parquet(f"{RAW}/seats.parquet", filters=[("hand_id", "in", hids)])
Ac = pd.read_parquet(f"{RAW}/actions.parquet", filters=[("hand_id", "in", hids)])
for tag, r in pick:
    h = H[H.hand_id == r.hand_id].iloc[0]; s = S[S.hand_id == r.hand_id].sort_values("seat_no"); a = Ac[Ac.hand_id == r.hand_id].sort_values("action_no")
    nm = {r.pa: "A", r.pb: "B"}
    print(f"\n##### {tag}  score {r.sc_fam:.3f}  hand {r.hand_id}  bb {h.big_blind}  board [{h.board_cards}]  pot {h.final_pot}  sd {h.players_at_showdown}")
    for x in s.itertuples():
        print(f"   seat{x.seat_no} {nm.get(x.player_id, '.')} {x.hole_card_1}{x.hole_card_2} stack {x.starting_stack} contrib {x.total_contribution} net {x.net_chips:+d} folded {int(x.folded)} sd {int(x.went_to_showdown)}")
    seat_of = dict(zip(s.player_id, s.seat_no))
    print("   " + " | ".join(f"{x.street[:2]} s{seat_of[x.player_id]}{nm.get(x.player_id, '')} {x.action}{'' if x.amount == 0 else ' ' + str(x.amount)}(tc{x.to_call})" for x in a.itertuples()))
