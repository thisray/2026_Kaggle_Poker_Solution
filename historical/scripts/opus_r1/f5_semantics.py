from common import *
con = connect()
Q(con,"SELECT street, action, count(*) n, avg(amount) amt, avg(to_call) tc, count(*) FILTER (amount>to_call) over_tc, count(*) FILTER (amount<to_call) under_tc, count(*) FILTER (amount=stack_before) eq_stack FROM actions GROUP BY ALL ORDER BY ALL")
Q(con,"SELECT count(*) n, count(*) FILTER (length(board_cards)=0) b0, count(*) FILTER (length(board_cards)=8) b3, count(*) FILTER (length(board_cards)=11) b4, count(*) FILTER (length(board_cards)=14) b5 FROM hands")
Q(con,"SELECT players_dealt, count(*) FROM hands GROUP BY 1")
Q(con,"SELECT count(*) n, sum(net_chips) s, count(*) FILTER (won_share>0 AND won_share<1) split FROM seats")
Q(con,"SELECT h.hand_id, sum(s.net_chips) snet, sum(s.total_contribution) sc, max(h.final_pot) fp FROM seats s JOIN hands h USING(hand_id) GROUP BY 1 HAVING sum(s.net_chips)<>0 LIMIT 5")
Q(con,"SELECT count(*) FROM (SELECT h.hand_id FROM seats s JOIN hands h USING(hand_id) GROUP BY 1 HAVING sum(s.net_chips)<>0)")
Q(con,"SELECT count(*) FROM (SELECT h.hand_id FROM seats s JOIN hands h USING(hand_id) GROUP BY 1 HAVING sum(s.total_contribution)<>max(h.final_pot))")
Q(con,"SELECT small_blind, big_blind, count(*) FROM hands GROUP BY ALL ORDER BY 3 DESC LIMIT 10")
Q(con,"SELECT preferred_stake, count(*) FROM players GROUP BY 1")
Q(con,"SELECT experience_hands_bucket, count(*) FROM players GROUP BY 1")
Q(con,"SELECT client_family, region_bucket, count(*) FROM players GROUP BY ALL ORDER BY ALL")
# stakes per table constant?
Q(con,"SELECT count(*) FILTER (nbb>1) tables_multi_bb FROM (SELECT table_id, count(DISTINCT big_blind) nbb FROM hands GROUP BY 1)")
# starting stacks relative to bb
Q(con,"SELECT quantile_cont(s.starting_stack::DOUBLE/h.big_blind,[0.01,0.1,0.5,0.9,0.99]) FROM seats s JOIN hands h USING(hand_id) USING SAMPLE 1%")
# seat rotation: is button_seat sequential? seats fixed per session?
Q(con,"SELECT hand_id, table_id, started_at, button_seat FROM hands WHERE table_id=(SELECT min(table_id) FROM hands) ORDER BY started_at LIMIT 12")
Q(con,"""WITH x AS (SELECT h.table_id, h.started_at, s.seat_no, s.player_id FROM hands h JOIN seats s USING(hand_id) WHERE table_id=(SELECT min(table_id) FROM hands))
SELECT started_at, string_agg(seat_no||':'||right(player_id,4), ' ' ORDER BY seat_no) seats FROM x GROUP BY 1 ORDER BY 1 LIMIT 25""")
# time gaps between hands in a table
Q(con,"""WITH x AS (SELECT table_id, started_at, epoch(started_at) - lag(epoch(started_at)) OVER (PARTITION BY table_id ORDER BY started_at) gap FROM hands)
SELECT quantile_cont(gap,[0.01,0.1,0.5,0.9,0.99,0.999]) FROM x""")
