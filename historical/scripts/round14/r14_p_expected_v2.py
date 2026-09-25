"""Round-14 P-line v2: phase-safe, act-order-correct conditional opportunity features.

Fixes vs r12_p_expected:
 - GROUP BY includes phase (no dev/eval mixing)
 - first/second roles use act_idx order (not player index)
 - conditional residual R = sum y_first*(y_second - q_second), V = sum y_first*q2*(1-q2)
 - sparse-activity log Bayes factor for fixed delta/rho grid
"""
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

OP = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
S = Path("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round11_scoped")
OUT = S / "r14_pexp_v2"; OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

ent = pd.read_parquet(S / "entry_decisions.parquet",
                      columns=["h", "player", "pool", "phase", "position", "faced_raise", "entered", "p_enter", "act_idx", "fold"])
ent["y"] = ent.entered.astype(float)
log("entry decisions", ent.shape, "mean p_enter", round(float(ent.p_enter.mean()), 4))
pdev = pd.read_parquet(f"{OP}/ptab_dev.parquet", columns=["pool", "p_lo", "p_hi"]); pdev["phase"] = 0
peval = pd.read_parquet(f"{OP}/ptab_eval.parquet", columns=["pool", "p_lo", "p_hi"]); peval["phase"] = 1
pairs = pd.concat([pdev, peval], ignore_index=True)

con = duckdb.connect(); con.execute("PRAGMA threads=8")
con.register("ent", ent); con.register("pairs", pairs)
con.execute("CREATE TABLE dec AS SELECT h, player, pool, phase, position, faced_raise, y, p_enter, act_idx FROM ent")
con.execute("CREATE TABLE pl AS SELECT pool, phase, p_lo AS plo, p_hi AS phi FROM pairs")
con.execute("""
CREATE TABLE ph AS
SELECT a.pool, a.phase, a.player AS plo, b.player AS phi, a.h,
  CASE WHEN a.act_idx <= b.act_idx THEN a.p_enter ELSE b.p_enter END AS p1,
  CASE WHEN a.act_idx <= b.act_idx THEN b.p_enter ELSE a.p_enter END AS p2,
  CASE WHEN a.act_idx <= b.act_idx THEN a.y ELSE b.y END AS y1,
  CASE WHEN a.act_idx <= b.act_idx THEN b.y ELSE a.y END AS y2
FROM dec a JOIN dec b ON a.h = b.h AND a.player < b.player
JOIN pl ON pl.pool = a.pool AND pl.phase = a.phase AND pl.plo = a.player AND pl.phi = b.player
""")
log("co-present rows", con.execute("SELECT count(*) FROM ph").fetchone()[0])
con.execute("""
CREATE TABLE agg AS
SELECT pool, phase, plo, phi,
  count(*) AS pe_n,
  sum(y1*y2) AS pe_obs_both, sum(p1*p2) AS pe_exp_both,
  sum(y1*(y2-p2)) AS pe_cond_resid, sum(y1*p2*(1-p2)) AS pe_cond_var,
  sum(y1) AS pe_n_first_entered, sum(y1*p2) AS pe_cond_exp,
  sum(CASE WHEN y1=1 THEN y2 ELSE 0 END) AS pe_obs_2g1
FROM ph GROUP BY pool, phase, plo, phi
""")
df = con.execute("SELECT * FROM agg").df()
log("agg", df.shape)
df["pe_rate_both"] = df.pe_obs_both / df.pe_n
df["pe_exp_rate_both"] = df.pe_exp_both / df.pe_n
df["pe_z_both"] = (df.pe_obs_both - df.pe_exp_both) / (df.pe_n * 0.25).clip(lower=0.25) ** 0.5
df["pe_cond_rate"] = df.pe_cond_resid / df.pe_n_first_entered.clip(lower=1)
df["pe_cond_z"] = df.pe_cond_resid / (df.pe_cond_var.clip(lower=1e-9)) ** 0.5
df["pe_cond_shrunk"] = df.pe_cond_resid / (df.pe_n_first_entered + 25.0)
df["pe_second_given_first"] = df.pe_obs_2g1 / df.pe_n_first_entered.clip(lower=1)
df["pe_cond_expected"] = df.pe_cond_exp / df.pe_n_first_entered.clip(lower=1)
df["pe_cond_gap"] = df.pe_second_given_first - df.pe_cond_expected

# sparse-activity log Bayes factor over fixed grids (entry event at second decision)
ph = con.execute("SELECT pool, phase, plo, phi, p2, y1, y2 FROM ph WHERE y1 = 1").df()
log("opportunity rows (y1=1)", ph.shape)
q = ph.p2.clip(1e-5, 1 - 1e-5).to_numpy(); yy = ph.y2.to_numpy(float)
lp = np.log(q / (1 - q))
for delta in [1.0, 2.0]:
    qd = 1 / (1 + np.exp(-(lp + delta)))
    ll = np.where(yy > 0, np.log(qd / q), np.log((1 - qd) / (1 - q)))
    for rho in [0.05, 0.1, 0.3]:
        name = f"pe_lbf_d{delta}_r{rho}"
        vals = np.logaddexp(np.log(1 - rho), np.log(rho) + ll)
        ph[name] = vals
lbf = ph.groupby(["pool", "phase", "plo", "phi"])[[c for c in ph.columns if c.startswith("pe_lbf")]].sum().reset_index()
df = df.merge(lbf, on=["pool", "phase", "plo", "phi"], how="left")
keep = ["pool", "phase", "plo", "phi"] + [c for c in df.columns if c.startswith("pe_")]
df[keep].to_parquet(OUT / "pair_entry_expected_v2.parquet")
log("saved", df.shape, "phase counts", df.phase.value_counts().to_dict())
print(df[[c for c in df.columns if c.startswith("pe_")]].describe().round(4).to_string())
