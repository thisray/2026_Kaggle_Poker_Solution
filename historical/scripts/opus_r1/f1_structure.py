import duckdb, json, sys
RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
con = duckdb.connect()
con.execute("SET threads=16")
for n, f in [("players","players.parquet"),("hands","hands.parquet"),("seats","seats.parquet"),("actions","actions.parquet")]:
    con.execute(f"CREATE VIEW {n} AS SELECT * FROM read_parquet('{RAW}/{f}')")
for n, f in [("labels","development_labels.csv"),("evidence","development_evidence.csv"),("evalp","evaluation_pairs.csv"),("sample","sample_submission.csv")]:
    con.execute(f"CREATE VIEW {n} AS SELECT * FROM read_csv_auto('{RAW}/{f}', header=true)")
def q(sql, n=50):
    df = con.execute(sql).df()
    print(df.head(n).to_string())
    return df
print("== schemas")
for t in ["players","hands","seats","actions","labels","evidence","evalp","sample"]:
    pass #print(t, con.execute(f"DESCRIBE {t}").df()[["column_name","column_type"]].values.tolist())
print("== sample rows")
#
#
#
#
#
#
#
print("== pool structure")
con.execute("CREATE TABLE pp AS SELECT s.player_id, min(h.table_id) pool, count(DISTINCT h.table_id) npools, count(*) nh, count(*) FILTER (h.phase='development') nh_dev, count(*) FILTER (h.phase='evaluation') nh_eval FROM seats s JOIN hands h USING(hand_id) GROUP BY 1")
q("SELECT max(npools) maxpools, count(*) players, count(DISTINCT pool) pools FROM pp")
q("SELECT pool, count(*) nplayers FROM pp GROUP BY 1 ORDER BY 2 LIMIT 3")
q("SELECT quantile_cont(nh,[0,0.01,0.1,0.5,0.9,0.99,1]) nh_q, quantile_cont(nh_dev,[0,0.1,0.5,0.9,1]) dev_q, quantile_cont(nh_eval,[0,0.1,0.5,0.9,1]) eval_q FROM pp")
q("SELECT phase, count(*) hands, count(DISTINCT table_id) ntables, min(started_at), max(started_at) FROM hands GROUP BY 1")
q("SELECT table_id, count(*) n, count(*) FILTER (phase='development') dev, count(*) FILTER (phase='evaluation') ev FROM hands GROUP BY 1 ORDER BY n LIMIT 3")
q("SELECT quantile_cont(n,[0,0.5,1]) FROM (SELECT table_id, count(*) n FROM hands GROUP BY 1)")
print("== labels")
q("SELECT label, label_status, behavior_family, count(*) FROM labels GROUP BY ALL ORDER BY ALL")
con.execute("CREATE TABLE lab AS SELECT l.*, p1.pool pool1, p2.pool pool2 FROM labels l JOIN pp p1 ON p1.player_id=l.player_1 JOIN pp p2 ON p2.player_id=l.player_2")
q("SELECT count(*) FILTER (pool1<>pool2) crosspool FROM lab")
q("SELECT npos, count(*) pools FROM (SELECT pool1, count(*) FILTER (label=1) npos FROM lab GROUP BY 1) GROUP BY 1 ORDER BY 1")
q("SELECT nlab, count(*) pools FROM (SELECT pool1, count(*) nlab FROM lab GROUP BY 1) GROUP BY 1 ORDER BY 1")
q("SELECT count(DISTINCT pool1) pools_with_labels FROM lab")
q("SELECT count(DISTINCT pool1) pools_with_pos FROM lab WHERE label=1")
# player multiplicity among positives
q("""WITH pl AS (SELECT player_1 p FROM labels WHERE label=1 UNION ALL SELECT player_2 FROM labels WHERE label=1)
SELECT cnt, count(*) FROM (SELECT p, count(*) cnt FROM pl GROUP BY 1) GROUP BY 1 ORDER BY 1""")
q("""WITH pl AS (SELECT player_1 p, label FROM labels UNION ALL SELECT player_2, label FROM labels)
SELECT label, cnt, count(*) FROM (SELECT p, label, count(*) cnt FROM pl GROUP BY 1,2) GROUP BY 1,2 ORDER BY 1,2""")
# do negatives involve positive players?
q("""WITH posp AS (SELECT player_1 p FROM labels WHERE label=1 UNION SELECT player_2 FROM labels WHERE label=1)
SELECT label, count(*) n, count(*) FILTER (player_1 IN (SELECT p FROM posp) OR player_2 IN (SELECT p FROM posp)) touches_pos FROM labels GROUP BY 1""")
# family per pool
q("SELECT fams, count(*) FROM (SELECT pool1, string_agg(DISTINCT behavior_family, '|' ORDER BY behavior_family) fams FROM lab WHERE label=1 GROUP BY 1) GROUP BY 1 ORDER BY 2 DESC")
print("== evidence")
q("SELECT count(*) rows, count(DISTINCT pair_id) pairs, count(DISTINCT hand_id) hands FROM evidence")
q("SELECT k, count(*) FROM (SELECT pair_id, count(*) k FROM evidence GROUP BY 1) GROUP BY 1 ORDER BY 1")
q("SELECT evidence_rank, count(*) FROM evidence GROUP BY 1 ORDER BY 1")
q("SELECT e.behavior_family, l.behavior_family lf, count(*) FROM evidence e JOIN labels l USING(pair_id) GROUP BY ALL")
q("SELECT h.phase, count(*) FROM evidence e JOIN hands h USING(hand_id) GROUP BY 1")
q("SELECT count(*) FROM labels l WHERE label=1 AND pair_id NOT IN (SELECT pair_id FROM evidence)")
print("== evaluation pairs")
q("SELECT count(*) n, count(DISTINCT pair_id), quantile_cont(shared_hands,[0,0.01,0.1,0.25,0.5,0.75,0.9,0.99,1]) q FROM evalp")
con.execute("CREATE TABLE ev AS SELECT e.*, p1.pool pool1, p2.pool pool2 FROM evalp e JOIN pp p1 ON p1.player_id=e.player_1 JOIN pp p2 ON p2.player_id=e.player_2")
q("SELECT count(*) FILTER (pool1<>pool2) crosspool, quantile_cont(n,[0,0.1,0.5,0.9,1]) per_pool FROM (SELECT pool1, pool2, count(*) OVER (PARTITION BY pool1) n FROM ev)")
q("SELECT count(*) FROM ev WHERE pair_id IN (SELECT pair_id FROM labels)")
q("""WITH posp AS (SELECT player_1 p FROM labels WHERE label=1 UNION SELECT player_2 FROM labels WHERE label=1)
SELECT count(*) FILTER (player_1 IN (SELECT p FROM posp) OR player_2 IN (SELECT p FROM posp)) touches_pos FROM evalp""")
q("SELECT min(shared_hands) FROM evalp")
q("SELECT player_1 < player_2 ordered, count(*) FROM evalp GROUP BY 1")
q("SELECT player_1 < player_2 ordered, count(*) FROM labels GROUP BY 1")
