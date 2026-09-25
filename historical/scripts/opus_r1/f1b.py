from common import *
con = connect()
con.execute("CREATE TABLE pp AS SELECT s.player_id, min(h.table_id) pool, count(*) nh, count(*) FILTER (h.phase='development') nh_dev, count(*) FILTER (h.phase='evaluation') nh_eval FROM seats s JOIN hands h USING(hand_id) GROUP BY 1")
print("== evidence")
Q(con,"SELECT count(*) nrows, count(DISTINCT pair_id) npairs, count(DISTINCT hand_id) nhands FROM evidence")
Q(con,"SELECT k, count(*) c FROM (SELECT pair_id, count(*) k FROM evidence GROUP BY 1) GROUP BY 1 ORDER BY 1")
Q(con,"SELECT evidence_rank, count(*) c FROM evidence GROUP BY 1 ORDER BY 1")
Q(con,"SELECT e.behavior_family ef, l.behavior_family lf, count(*) c FROM evidence e JOIN labels l USING(pair_id) GROUP BY ALL")
Q(con,"SELECT h.phase, count(*) c FROM evidence e JOIN hands h USING(hand_id) GROUP BY 1")
Q(con,"SELECT count(*) c FROM labels l WHERE label=1 AND pair_id NOT IN (SELECT pair_id FROM evidence)")
Q(con,"SELECT l.behavior_family, k, count(*) c FROM (SELECT pair_id, count(*) k FROM evidence GROUP BY 1) e JOIN labels l USING(pair_id) GROUP BY ALL ORDER BY ALL")
# hands shared by multiple positive pairs as evidence
Q(con,"SELECT m, count(*) c FROM (SELECT hand_id, count(*) m FROM evidence GROUP BY 1) GROUP BY 1")
print("== evaluation pairs")
Q(con,"SELECT count(*) n, count(DISTINCT pair_id) nd, quantile_cont(shared_hands,[0,0.01,0.1,0.25,0.5,0.75,0.9,0.99,1]) q FROM evalp")
con.execute("CREATE TABLE ev AS SELECT e.*, p1.pool pool1, p2.pool pool2 FROM evalp e JOIN pp p1 ON p1.player_id=e.player_1 JOIN pp p2 ON p2.player_id=e.player_2")
Q(con,"SELECT count(*) FILTER (pool1<>pool2) crosspool FROM ev")
Q(con,"SELECT quantile_cont(n,[0,0.1,0.5,0.9,1]) per_pool FROM (SELECT pool1, count(*) n FROM ev GROUP BY 1)")
Q(con,"SELECT count(*) c FROM ev WHERE pair_id IN (SELECT pair_id FROM labels)")
Q(con,"""WITH posp AS (SELECT player_1 p FROM labels WHERE label=1 UNION SELECT player_2 FROM labels WHERE label=1)
SELECT count(*) FILTER (player_1 IN (SELECT p FROM posp) OR player_2 IN (SELECT p FROM posp)) touches_pos FROM evalp""")
Q(con,"SELECT (player_1 < player_2) ordered, count(*) c FROM evalp GROUP BY 1")
Q(con,"SELECT (player_1 < player_2) ordered, count(*) c FROM labels GROUP BY 1")
# full pair universe in eval phase: which pairs co-occur, and are they in evalp?
con.execute("""CREATE TABLE cooc AS
SELECT least(s1.player_id,s2.player_id) p1, greatest(s1.player_id,s2.player_id) p2, h.table_id pool,
 count(*) FILTER (h.phase='development') sh_dev, count(*) FILTER (h.phase='evaluation') sh_eval
FROM seats s1 JOIN seats s2 ON s1.hand_id=s2.hand_id AND s1.player_id<s2.player_id JOIN hands h ON h.hand_id=s1.hand_id
GROUP BY ALL""")
Q(con,"SELECT count(*) n_cooc_pairs, count(*) FILTER (sh_eval>0) n_eval_cooc, count(*) FILTER (sh_dev>0) n_dev_cooc FROM cooc")
con.execute("""CREATE TABLE posp AS SELECT player_1 p FROM labels WHERE label=1 UNION SELECT player_2 FROM labels WHERE label=1""")
con.execute("""CREATE TABLE univ AS SELECT c.*, 
  (SELECT count(*) FROM labels l WHERE least(l.player_1,l.player_2)=c.p1 AND greatest(l.player_1,l.player_2)=c.p2) is_lab,
  (c.p1 IN (SELECT p FROM posp) OR c.p2 IN (SELECT p FROM posp)) touches_pos
FROM cooc c""")
con.execute("""CREATE TABLE evn AS SELECT least(player_1,player_2) p1, greatest(player_1,player_2) p2, shared_hands FROM evalp""")
Q(con,"""SELECT u.is_lab, u.touches_pos, (e.p1 IS NOT NULL) in_eval, count(*) c, min(u.sh_eval) mn, quantile_cont(u.sh_eval,0.5) med
FROM univ u LEFT JOIN evn e ON e.p1=u.p1 AND e.p2=u.p2 GROUP BY ALL ORDER BY ALL""")
Q(con,"SELECT count(*) c FROM evn e JOIN univ u ON e.p1=u.p1 AND e.p2=u.p2 WHERE e.shared_hands<>u.sh_eval")
# exposure: expected co-occurrence given activity
Q(con,"""WITH x AS (SELECT u.*, a.nh_dev a_dev, b.nh_dev b_dev, a.nh_eval a_ev, b.nh_eval b_ev FROM univ u JOIN pp a ON a.player_id=u.p1 JOIN pp b ON b.player_id=u.p2)
SELECT corr(sh_dev, a_dev*b_dev) c_dev, corr(sh_eval, a_ev*b_ev) c_ev, avg(sh_dev/(a_dev*b_dev/3000.0)) ratio_dev, avg(sh_eval/(a_ev*b_ev/2000.0)) ratio_ev FROM x""")
