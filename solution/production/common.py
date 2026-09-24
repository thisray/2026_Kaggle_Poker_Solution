import duckdb, pandas as pd, numpy as np
RAW = __import__("os").environ["POKER_DATA_DIR"]
OUT = __import__("os").environ["POKER_WORK_DIR"]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_colwidth", 80)
def connect(threads=16, mem="90GB"):
    con = duckdb.connect()
    con.execute(f"SET threads={threads}"); con.execute(f"SET memory_limit='{mem}'")
    con.execute("SET preserve_insertion_order=false")
    for n, f in [("players","players.parquet"),("hands","hands.parquet"),("seats","seats.parquet"),("actions","actions.parquet")]:
        con.execute(f"CREATE VIEW {n} AS SELECT * FROM read_parquet('{RAW}/{f}')")
    for n, f in [("labels","development_labels.csv"),("evidence","development_evidence.csv"),("evalp","evaluation_pairs.csv"),("sample","sample_submission.csv")]:
        con.execute(f"CREATE VIEW {n} AS SELECT * FROM read_csv_auto('{RAW}/{f}', header=true)")
    return con
def Q(con, sql, n=60, show=True):
    df = con.execute(sql).df()
    if show: print(df.head(n).to_string()); print()
    return df
