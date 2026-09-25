from common import *
import sys
con = connect()
fam = sys.argv[1]; npairs = int(sys.argv[2]); seed = int(sys.argv[3])
pairs = Q(con, f"SELECT pair_id, player_1, player_2 FROM labels WHERE behavior_family='{fam}' ORDER BY hash(pair_id || '{seed}') LIMIT {npairs}", show=False)
for _, pr in pairs.iterrows():
    A, B = pr.player_1, pr.player_2
    ev = Q(con, f"SELECT e.evidence_rank, e.hand_id FROM evidence e WHERE pair_id='{pr.pair_id}' ORDER BY evidence_rank", show=False)
    print("#"*100); print("PAIR", pr.pair_id, fam, "A=",A,"B=",B)
    for _, e in ev.iterrows():
        h = con.execute(f"SELECT hand_id, started_at, button_seat, small_blind sb, big_blind bb, board_cards, final_pot, players_dealt, players_at_showdown FROM hands WHERE hand_id='{e.hand_id}'").df().iloc[0]
        s = con.execute(f"SELECT player_id, seat_no, starting_stack st, hole_card_1||hole_card_2 hole, total_contribution contrib, net_chips net, folded, went_to_showdown sd, won_share ws FROM seats WHERE hand_id='{e.hand_id}' ORDER BY seat_no").df()
        a = con.execute(f"SELECT action_no, street, player_id, action, amount, amount_to, pot_before, stack_before, to_call, players_active FROM actions WHERE hand_id='{e.hand_id}' ORDER BY action_no").df()
        tag = {A:"A", B:"B"}
        s["who"] = s.player_id.map(lambda p: tag.get(p, "."))
        a["who"] = a.player_id.map(lambda p: tag.get(p, "o" + p[-3:]))
        s["player_id"] = s.player_id.map(lambda p: tag.get(p, "o" + p[-3:]))
        print(f"--- ev_rank={e.evidence_rank} hand={h.hand_id} t={h.started_at} btn={h.button_seat} blinds={h.sb}/{h.bb} board=[{h.board_cards}] pot={h.final_pot} dealt={h.players_dealt} sd={h.players_at_showdown}")
        print(s.drop(columns=["who"]).to_string(index=False))
        print(a.drop(columns=["player_id"]).to_string(index=False))
