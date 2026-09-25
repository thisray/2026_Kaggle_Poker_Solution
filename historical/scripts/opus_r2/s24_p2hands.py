"""Raw hands of eval 'pattern-2' pairs (both raise more / fold less preflop when partner holds a strong hand, before partner acts)."""
import numpy as np, pandas as pd, sys
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; RAW = f"{A_}/data/raw"
ev = pd.read_csv(f"{RAW}/evaluation_pairs.csv"); rk = pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv")
slots = [int(x) for x in sys.argv[1].split(",")]; nshow = int(sys.argv[2]) if len(sys.argv) > 2 else 6
rk = rk[rk.slot.isin(slots)].merge(ev, on="pair_id")
H = pd.read_parquet(f"{RAW}/hands.parquet", columns=["hand_id", "started_at", "phase", "big_blind", "board_cards", "final_pot", "players_at_showdown"])
S = pd.read_parquet(f"{RAW}/seats.parquet")
for r in rk.itertuples():
    a, b = r.player_1, r.player_2
    hs = S[S.player_id.isin([a, b])].groupby("hand_id").player_id.nunique(); hs = hs[hs == 2].index
    hh = H[H.hand_id.isin(hs) & (H.phase != "development")].sort_values("started_at")
    ss = S[S.hand_id.isin(hh.hand_id)]
    Ac = pd.read_parquet(f"{RAW}/actions.parquet", filters=[("hand_id", "in", hh.hand_id.tolist())])
    # pick hands where a pair member raised preflop and the partner is still in the hand after preflop
    pf = Ac[(Ac.street == "preflop")]
    rows = []
    for hid, g in pf.groupby("hand_id"):
        ra = g[g.player_id.isin([a, b]) & g.action.isin(["raise", "bet", "all_in"])]
        if len(ra) and g[g.player_id.isin([a, b])].player_id.nunique() == 2: rows.append(hid)
    print(f"\n######## slot {r.slot} pair {r.pair_id} risk {r.risk_score:.3f} shared {r.shared_hands}  hands both-in-preflop-with-pair-raise: {len(rows)}")
    for hid in rows[:nshow]:
        h = hh[hh.hand_id == hid].iloc[0]; s = ss[ss.hand_id == hid].sort_values("seat_no"); act = Ac[Ac.hand_id == hid].sort_values("action_no")
        nm = {a: "A", b: "B"}; seat_of = dict(zip(s.player_id, s.seat_no))
        print(f"  -- {hid} bb {h.big_blind} board [{h.board_cards}] pot {h.final_pot} sd {h.players_at_showdown} | " + " ".join(f"s{x.seat_no}{nm.get(x.player_id, '')}:{x.hole_card_1}{x.hole_card_2}({x.net_chips:+d})" for x in s.itertuples()))
        print("     " + " | ".join(f"{x.street[:2]} s{seat_of[x.player_id]}{nm.get(x.player_id, '')} {x.action}{'' if x.amount == 0 else ' ' + str(x.amount)}" for x in act.itertuples()))
