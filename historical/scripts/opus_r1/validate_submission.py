"""Independent legality check: every evidence hand is an evaluation-phase hand where both players of the pair were seated."""
import duckdb, sys
RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
path = sys.argv[1]
con = duckdb.connect(); con.execute("SET threads=16")
con.execute(f"CREATE VIEW sub AS SELECT * FROM read_csv_auto('{path}', header=true)")
con.execute(f"CREATE VIEW evalp AS SELECT * FROM read_csv_auto('{RAW}/evaluation_pairs.csv', header=true)")
con.execute(f"CREATE VIEW seats AS SELECT hand_id, player_id FROM read_parquet('{RAW}/seats.parquet')")
con.execute(f"CREATE VIEW hands AS SELECT hand_id, phase FROM read_parquet('{RAW}/hands.parquet')")
con.execute("""CREATE TABLE ev AS SELECT pair_id, unnest([evidence_hand_1,evidence_hand_2,evidence_hand_3,evidence_hand_4,evidence_hand_5]) hand_id FROM sub""")
print(con.execute("SELECT count(*) n, count(*) FILTER (hand_id='NO_EVIDENCE') noev FROM ev").fetchall())
bad = con.execute("""SELECT count(*) FROM ev e JOIN evalp p USING(pair_id)
 LEFT JOIN hands h ON h.hand_id=e.hand_id
 LEFT JOIN seats s1 ON s1.hand_id=e.hand_id AND s1.player_id=p.player_1
 LEFT JOIN seats s2 ON s2.hand_id=e.hand_id AND s2.player_id=p.player_2
 WHERE e.hand_id<>'NO_EVIDENCE' AND (h.phase IS NULL OR h.phase<>'evaluation' OR s1.player_id IS NULL OR s2.player_id IS NULL)""").fetchall()
print("illegal evidence cells:", bad)
print("pairs:", con.execute("SELECT count(*), count(DISTINCT pair_id) FROM sub").fetchall(), "missing vs evalp:", con.execute("SELECT count(*) FROM evalp WHERE pair_id NOT IN (SELECT pair_id FROM sub)").fetchall())
print("risk range:", con.execute("SELECT min(risk_score), max(risk_score), count(*) FILTER (risk_score IS NULL) FROM sub").fetchall())
