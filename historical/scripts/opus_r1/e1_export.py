"""Export raw tables into compact numpy arrays (hand-ordered) for numba kernels."""
from common import *
import time, os
t0 = time.time()
con = connect()
D = f"{OUT}/np"; os.makedirs(D, exist_ok=True)
card_rank = "CASE substr(c,1,1) WHEN '2' THEN 0 WHEN '3' THEN 1 WHEN '4' THEN 2 WHEN '5' THEN 3 WHEN '6' THEN 4 WHEN '7' THEN 5 WHEN '8' THEN 6 WHEN '9' THEN 7 WHEN 'T' THEN 8 WHEN 'J' THEN 9 WHEN 'Q' THEN 10 WHEN 'K' THEN 11 WHEN 'A' THEN 12 END"
card_suit = "CASE substr(c,2,1) WHEN 'c' THEN 0 WHEN 'd' THEN 1 WHEN 'h' THEN 2 WHEN 's' THEN 3 END"
con.execute(f"CREATE MACRO card(c) AS ({card_rank})*4 + ({card_suit})")
con.execute("CREATE TABLE pidx AS SELECT player_id, (row_number() OVER (ORDER BY player_id))-1 AS pi FROM players")
con.execute("CREATE TABLE tidx AS SELECT table_id, (row_number() OVER (ORDER BY table_id))-1 AS ti FROM (SELECT DISTINCT table_id FROM hands)")
con.execute("""CREATE TABLE hidx AS SELECT h.hand_id, (row_number() OVER (ORDER BY t.ti, h.started_at, h.hand_id))-1 AS hi, t.ti,
  epoch(h.started_at) ts, (h.phase='evaluation')::INT ph, h.small_blind sb, h.big_blind bb, h.final_pot pot, h.button_seat btn,
  h.players_at_showdown nsd, h.board_cards FROM hands h JOIN tidx t USING(table_id)""")
H = con.execute("SELECT * FROM hidx ORDER BY hi").df()
print("hands", len(H), time.time()-t0)
np.save(f"{D}/h_table.npy", H.ti.values.astype(np.int32))
np.save(f"{D}/h_ts.npy", H.ts.values.astype(np.float64))
np.save(f"{D}/h_phase.npy", H.ph.values.astype(np.int8))
np.save(f"{D}/h_sb.npy", H.sb.values.astype(np.int32)); np.save(f"{D}/h_bb.npy", H.bb.values.astype(np.int32))
np.save(f"{D}/h_pot.npy", H.pot.values.astype(np.int32)); np.save(f"{D}/h_btn.npy", H.btn.values.astype(np.int8)); np.save(f"{D}/h_nsd.npy", H.nsd.values.astype(np.int8))
rmap = {r:i for i,r in enumerate("23456789TJQKA")}; smap = {s:i for i,s in enumerate("cdhs")}
board = np.full((len(H),5), -1, np.int16)
for k, bc in enumerate(H.board_cards.values):
    if bc:
        for j, c in enumerate(bc.split()):
            board[k,j] = rmap[c[0]]*4 + smap[c[1]]
np.save(f"{D}/h_board.npy", board)
H[["hand_id","hi"]].to_parquet(f"{D}/hand_index.parquet")
con.execute("SELECT * FROM pidx").df().to_parquet(f"{D}/player_index.parquet")
con.execute("SELECT * FROM tidx").df().to_parquet(f"{D}/table_index.parquet")
print("board done", time.time()-t0)
S = con.execute("""SELECT x.hi, s.seat_no, p.pi, s.starting_stack, card(s.hole_card_1) c1, card(s.hole_card_2) c2, s.total_contribution, s.net_chips,
   s.folded::INT folded, s.went_to_showdown::INT sd, s.won_share FROM seats s JOIN hidx x USING(hand_id) JOIN pidx p USING(player_id) ORDER BY x.hi, s.seat_no""").df()
assert len(S) == 6*len(H)
assert (S.seat_no.values.reshape(-1,6) == np.arange(6)).all()
def r6(c, dt): return S[c].values.astype(dt).reshape(-1,6)
np.save(f"{D}/s_player.npy", r6("pi", np.int32)); np.save(f"{D}/s_stack.npy", r6("starting_stack", np.int32))
np.save(f"{D}/s_c1.npy", r6("c1", np.int16)); np.save(f"{D}/s_c2.npy", r6("c2", np.int16))
np.save(f"{D}/s_contrib.npy", r6("total_contribution", np.int32)); np.save(f"{D}/s_net.npy", r6("net_chips", np.int32))
np.save(f"{D}/s_folded.npy", r6("folded", np.int8)); np.save(f"{D}/s_sd.npy", r6("sd", np.int8)); np.save(f"{D}/s_won.npy", r6("won_share", np.float32))
print("seats done", time.time()-t0)
del S
A = con.execute("""SELECT x.hi, a.action_no, CASE a.street WHEN 'preflop' THEN 0 WHEN 'flop' THEN 1 WHEN 'turn' THEN 2 ELSE 3 END st,
   s.seat_no seat, CASE a.action WHEN 'fold' THEN 0 WHEN 'check' THEN 1 WHEN 'call' THEN 2 WHEN 'bet' THEN 3 WHEN 'raise' THEN 4 WHEN 'all_in' THEN 5 END act,
   a.amount, a.amount_to, a.pot_before, a.stack_before, a.to_call, a.players_active
 FROM actions a JOIN hidx x USING(hand_id) JOIN seats s ON s.hand_id=a.hand_id AND s.player_id=a.player_id ORDER BY x.hi, a.action_no""").df()
print("actions", len(A), time.time()-t0)
hi = A.hi.values.astype(np.int64)
off = np.zeros(len(H)+1, np.int64); np.add.at(off, hi+1, 1); off = np.cumsum(off)
np.save(f"{D}/a_off.npy", off)
for c, dt in [("action_no",np.int16),("st",np.int8),("seat",np.int8),("act",np.int8),("amount",np.int32),("amount_to",np.int32),("pot_before",np.int32),("stack_before",np.int32),("to_call",np.int32),("players_active",np.int8)]:
    np.save(f"{D}/a_{c}.npy", A[c].values.astype(dt))
print("done", time.time()-t0)
