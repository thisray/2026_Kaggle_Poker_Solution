from common import *
import time
t0=time.time()
con = connect()
# labelled pairs, dev phase shared hands
con.execute("""CREATE TABLE lp AS SELECT pair_id, player_1 A, player_2 B, label, behavior_family fam FROM labels""")
con.execute("""CREATE TABLE ph AS
SELECT lp.pair_id, lp.label, lp.fam, h.hand_id, h.started_at, h.big_blind bb, h.final_pot, h.players_dealt, h.players_at_showdown, h.board_cards,
  sa.seat_no a_seat, sb.seat_no b_seat, sa.starting_stack a_stack, sb.starting_stack b_stack,
  sa.total_contribution a_contrib, sb.total_contribution b_contrib, sa.net_chips a_net, sb.net_chips b_net,
  sa.folded a_fold, sb.folded b_fold, sa.went_to_showdown a_sd, sb.went_to_showdown b_sd, sa.won_share a_ws, sb.won_share b_ws,
  sa.hole_card_1||sa.hole_card_2 a_hole, sb.hole_card_1||sb.hole_card_2 b_hole,
  (e.hand_id IS NOT NULL) is_ev, e.evidence_rank
FROM lp JOIN seats sa ON sa.player_id=lp.A JOIN seats sb ON sb.player_id=lp.B AND sb.hand_id=sa.hand_id
JOIN hands h ON h.hand_id=sa.hand_id AND h.phase='development'
LEFT JOIN evidence e ON e.pair_id=lp.pair_id AND e.hand_id=h.hand_id""")
Q(con,"SELECT label, fam, count(*) n, count(*) FILTER (is_ev) nev FROM ph GROUP BY ALL ORDER BY ALL")
# action-level: tag actor who
con.execute("""CREATE TABLE pa AS
SELECT ph.pair_id, ph.hand_id, a.action_no, a.street, a.action, a.amount, a.amount_to, a.pot_before, a.stack_before, a.to_call, a.players_active,
  CASE WHEN a.player_id=lp.A THEN 'A' WHEN a.player_id=lp.B THEN 'B' ELSE 'O' END who
FROM ph JOIN lp USING(pair_id) JOIN actions a ON a.hand_id=ph.hand_id""")
# last aggressor before each action (by who), computed with window
con.execute("""CREATE TABLE pa2 AS
SELECT *, 
  last_value(CASE WHEN action IN ('bet','raise') OR (action='all_in' AND amount>to_call) THEN who END IGNORE NULLS) OVER (PARTITION BY pair_id, hand_id ORDER BY action_no ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) last_aggr,
  (action IN ('bet','raise') OR (action='all_in' AND amount>to_call)) is_aggr
FROM pa""")
con.execute("""CREATE TABLE hf AS
SELECT pair_id, hand_id,
  count(*) FILTER (who='A') a_n, count(*) FILTER (who='B') b_n,
  count(*) FILTER (who='A' AND is_aggr) a_aggr, count(*) FILTER (who='B' AND is_aggr) b_aggr,
  count(*) FILTER (who='O' AND is_aggr) o_aggr,
  count(*) FILTER (who='A' AND action='fold' AND last_aggr='B') a_fold_to_b,
  count(*) FILTER (who='B' AND action='fold' AND last_aggr='A') b_fold_to_a,
  count(*) FILTER (who='A' AND action='fold' AND last_aggr='O') a_fold_to_o,
  count(*) FILTER (who='B' AND action='fold' AND last_aggr='O') b_fold_to_o,
  count(*) FILTER (who='O' AND action='fold' AND last_aggr IN ('A','B')) o_fold_to_ab,
  count(*) FILTER (who='A' AND action IN ('call') AND last_aggr='B') a_call_b,
  count(*) FILTER (who='B' AND action IN ('call') AND last_aggr='A') b_call_a,
  count(*) FILTER (who='A' AND is_aggr AND last_aggr='B') a_reraise_b,
  count(*) FILTER (who='B' AND is_aggr AND last_aggr='A') b_reraise_a,
  count(*) FILTER (who='A' AND is_aggr AND last_aggr='O') a_raise_o,
  count(*) FILTER (who='B' AND is_aggr AND last_aggr='O') b_raise_o,
  count(*) FILTER (who IN ('A','B') AND street<>'preflop' AND players_active=2 AND action='check') ab_hu_checks,
  count(*) FILTER (street<>'preflop' AND players_active=2 AND who IN ('A','B')) ab_hu_actions_any,
  max(CASE WHEN street='preflop' THEN 0 WHEN street='flop' THEN 1 WHEN street='turn' THEN 2 ELSE 3 END) max_street,
  bool_or(who='A' AND action<>'fold' AND street='preflop' AND amount>0) a_vpip,
  bool_or(who='B' AND action<>'fold' AND street='preflop' AND amount>0) b_vpip,
  max(amount) FILTER (who='A') a_max_amt, max(amount) FILTER (who='B') b_max_amt
FROM pa2 GROUP BY 1,2""")
con.execute("""CREATE TABLE phf AS SELECT ph.*, hf.* EXCLUDE (pair_id, hand_id),
  (a_net::DOUBLE/bb) a_net_bb, (b_net::DOUBLE/bb) b_net_bb, (a_contrib::DOUBLE/bb) a_c_bb, (b_contrib::DOUBLE/bb) b_c_bb,
  least(greatest(-a_net,0), greatest(b_net,0))::DOUBLE/bb t_ab, least(greatest(-b_net,0), greatest(a_net,0))::DOUBLE/bb t_ba,
  (NOT a_fold AND NOT b_fold) both_end
FROM ph JOIN hf USING(pair_id, hand_id)""")
con.execute(f"COPY phf TO '{OUT}/lab_pairhand_dev_v1.parquet' (FORMAT parquet)")
print("built", time.time()-t0)
df = con.execute("SELECT * FROM phf").df()
df["grp"] = np.where(df.label==0, "neg", np.where(df.is_ev, df.fam+"|EV", df.fam+"|non"))
# symmetric/directional: for positive pairs, orient donor as the one losing more in evidence hands (directed) -- first just raw
cols = ["a_vpip","b_vpip","max_street","players_at_showdown","a_c_bb","b_c_bb","a_net_bb","b_net_bb","t_ab","t_ba","a_aggr","b_aggr","o_aggr",
        "a_fold_to_b","b_fold_to_a","a_fold_to_o","b_fold_to_o","o_fold_to_ab","a_call_b","b_call_a","a_reraise_b","b_reraise_a","a_raise_o","b_raise_o","ab_hu_checks","ab_hu_actions_any","both_end","a_sd","b_sd"]
d2 = df.copy()
for c in ["a_vpip","b_vpip","both_end","a_sd","b_sd"]: d2[c]=d2[c].astype(float)
print(d2.groupby("grp")[cols].mean().T.round(3).to_string())
print(d2.groupby("grp").size())
