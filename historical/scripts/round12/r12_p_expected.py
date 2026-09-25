"""Round-12 P-line: opportunity-normalized expected-count pair features (DuckDB).

For every pair in ptab_dev (phase 0) / ptab_eval (phase 1): over shared hands,
observed vs expected both-entry, first/second-role conditional deviation, residual
z with exposure shrinkage. Output: pair_entry_expected.parquet keyed (pool, plo, phi).
"""
import json
import time
from pathlib import Path

import duckdb
import pandas as pd

OP = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
S = Path("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round11_scoped")
OUT = S / "r12_pexp"; OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

ent = pd.read_parquet(S / "entry_decisions.parquet",
                      columns=["h", "player", "pool", "phase", "position", "faced_raise", "entered", "p_enter", "act_idx", "fold"])
ent["y"] = ent.entered.astype(float)
pdev = pd.read_parquet(f"{OP}/ptab_dev.parquet", columns=["pool", "p_lo", "p_hi"])
peval = pd.read_parquet(f"{OP}/ptab_eval.parquet", columns=["pool", "p_lo", "p_hi"])
pdev["phase"] = 0; peval["phase"] = 1
pairs = pd.concat([pdev, peval], ignore_index=True)
log("inputs", ent.shape, pairs.shape)

con = duckdb.connect()
con.execute("PRAGMA threads=8")
con.register("ent", ent)
con.register("pairs", pairs)
con.execute("CREATE TABLE dec AS SELECT h, player, pool, phase, position, faced_raise, y, p_enter, act_idx, fold FROM ent")
con.execute("CREATE TABLE pl AS SELECT pool, phase, p_lo AS plo, p_hi AS phi FROM pairs")
q = """
CREATE TABLE ph AS
SELECT a.pool, a.phase, a.player AS plo, b.player AS phi, a.h,
       a.p_enter AS p1, b.p_enter AS p2, a.y AS y1, b.y AS y2,
       a.act_idx AS a1, b.act_idx AS a2
FROM dec a
JOIN dec b ON a.h = b.h AND a.player < b.player
JOIN pl ON pl.pool = a.pool AND pl.phase = a.phase AND pl.plo = a.player AND pl.phi = b.player
"""
con.execute(q)
log("co-present rows", con.execute("SELECT count(*) FROM ph").fetchone()[0])
con.execute("""
CREATE TABLE agg AS
SELECT pool, plo, phi,
  count(*) AS pe_n,
  sum(y1*y2) AS pe_obs_both,
  sum(p1*p2) AS pe_exp_both,
  sum(y1*y2 - p1*p2) AS pe_resid_both,
  sum(p1*p2*(1-p1*p2)) AS pe_v_both,
  sum(y1) AS pe_obs_first,
  sum(p1) AS pe_exp_first,
  sum(y1-p1) AS pe_resid_first,
  sum(CASE WHEN y1=1 THEN y2 ELSE 0 END) AS pe_obs_2g1,
  sum(CASE WHEN y1=1 THEN p2 ELSE 0 END) AS pe_exp_2g1,
  sum(CASE WHEN y1=1 THEN y2-p2 ELSE 0 END) AS pe_resid_2g1,
  sum(y1) AS pe_n_second,
  first(phase) AS phase,
  min(pool) AS _p
FROM ph GROUP BY pool, plo, phi
""")
df = con.execute("""
SELECT pool, phase, plo, phi, pe_n, pe_obs_both, pe_exp_both, pe_resid_both, pe_v_both,
       pe_obs_first, pe_exp_first, pe_resid_first, pe_obs_2g1, pe_exp_2g1, pe_resid_2g1, pe_n_second
FROM agg
""").df()
log("agg", df.shape)
df["pe_rate_both"] = df.pe_obs_both / df.pe_n
df["pe_exp_rate_both"] = df.pe_exp_both / df.pe_n
df["pe_z_both"] = df.pe_resid_both / (df.pe_v_both.clip(lower=0) ** 0.5 + 1e-9)
df["pe_shrunk_resid"] = df.pe_resid_both / (df.pe_n + 25.0)
m2 = df.pe_n_second > 0
df["pe_second_given_first"] = (df.pe_obs_2g1 / df.pe_n_second.clip(lower=1))
df["pe_exp_second_given_first"] = (df.pe_exp_2g1 / df.pe_n_second.clip(lower=1))
df["pe_resid_second_given_first"] = (df.pe_resid_2g1 / df.pe_n_second.clip(lower=1))
df["pe_z_second_given_first"] = df.pe_resid_2g1 / (0.25 * df.pe_n_second.clip(lower=1)) ** 0.5
keep = ["pool", "phase", "plo", "phi"] + [c for c in df.columns if c.startswith("pe_")]
df[keep].to_parquet(OUT / "pair_entry_expected.parquet")
log("saved", df.shape)
print(df[[c for c in df.columns if c.startswith("pe_")]].describe().round(4).to_string())
