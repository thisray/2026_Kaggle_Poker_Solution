"""R4-U12: generator-artifact screen inside the R15 top-10 (the region that decides AP@5): hand duration, timestamp digits, bet-size ratios, amount roundness, stacks.
Univariate AUC of evidence vs non-evidence per family (conditional on being a top-10 candidate)."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
hi = pd.read_parquet(f"{D}/hand_index.parquet"); hmap = dict(zip(hi.hand_id, hi.hi))
t = pd.read_parquet(f"{OUT}/r3/t45_known_e_rerank.parquet"); t["h"] = t.hand_id.map(hmap); t["rk"] = t.groupby("slot").b.rank(ascending=False, method="first")
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy", mmap_mode="r"); sp = np.load(f"{D}/s_player.npy", mmap_mode="r"); a_act = np.load(f"{D}/a_act.npy", mmap_mode="r")
a_amt = np.load(f"{D}/a_amount.npy", mmap_mode="r"); a_to = np.load(f"{D}/a_amount_to.npy", mmap_mode="r"); a_pot = np.load(f"{D}/a_pot_before.npy", mmap_mode="r"); a_st = np.load(f"{D}/a_st.npy", mmap_mode="r")
a_stack = np.load(f"{D}/a_stack_before.npy", mmap_mode="r"); ts = np.load(f"{D}/h_ts.npy"); tab = np.load(f"{D}/h_table.npy"); bb = np.load(f"{D}/h_bb.npy"); stack = np.load(f"{D}/s_stack.npy", mmap_mode="r"); hpot = np.load(f"{D}/h_pot.npy")
# duration: next hand at the same table (pool)
order = np.lexsort((ts, tab)); nxt = np.full(len(ts), np.nan); same = tab[order][1:] == tab[order][:-1]; nxt[order[:-1][same]] = (ts[order][1:] - ts[order][:-1])[same]
prv = np.full(len(ts), np.nan); prv[order[1:][same]] = (ts[order][1:] - ts[order][:-1])[same]
rows = []
for h, pa, pb in zip(t.h.values, t.pa.values, t.pb.values):
    seats = np.asarray(sp[h]); sa = int(np.flatnonzero(seats == pa)[0]); sb = int(np.flatnonzero(seats == pb)[0])
    ks = np.arange(off[h], off[h + 1]); seat = np.asarray(a_seat[ks]); act = np.asarray(a_act[ks]); amt = np.asarray(a_amt[ks]).astype(float); pot = np.asarray(a_pot[ks]).astype(float); st = np.asarray(a_st[ks]); to = np.asarray(a_to[ks]).astype(float)
    mem = (seat == sa) | (seat == sb); B = float(bb[h]); na = len(ks)
    betm = mem & (act == 3); raism = mem & (act == 4) & (st == 0); beto = ~mem & (act == 3)
    r_bet = (amt[betm] / np.maximum(pot[betm], 1)); r_beto = (amt[beto] / np.maximum(pot[beto], 1))
    rows.append((nxt[h], prv[h], nxt[h] / max(na, 1), na, ts[h] % 1.0, (ts[h] * 1000) % 10, r_bet.mean() if len(r_bet) else np.nan, r_bet.max() if len(r_bet) else np.nan, r_bet.min() if len(r_bet) else np.nan,
                 r_beto.mean() if len(r_beto) else np.nan, (to[raism] / B).mean() if raism.any() else np.nan, float(np.mean((amt[mem & (act >= 2)] / B * 2) % 1 == 0)) if (mem & (act >= 2)).any() else np.nan,
                 stack[h, sa] / B, stack[h, sb] / B, abs(stack[h, sa] - stack[h, sb]) / B, float(stack[h, sa] / B == 100) + float(stack[h, sb] / B == 100), hpot[h] / B, (hpot[h] / B * 2) % 1))
cols = ["dur_next", "dur_prev", "dur_per_action", "n_actions", "ts_frac", "ts_ms_digit", "mem_bet_ratio_mean", "mem_bet_ratio_max", "mem_bet_ratio_min", "out_bet_ratio_mean", "mem_pf_raise_to", "mem_amt_halfbb_round", "stackA", "stackB", "stack_absdiff", "n_stack100", "pot_bb", "pot_halfbb_frac"]
for j, c in enumerate(cols): t[c] = [r[j] for r in rows]
x = t[t.rk <= 10]
res = []
for c in cols + ["b", "nd", "sur_c_max"]:
    r = {"feat": c, "cover": x[c].notna().mean()}
    for fam, g in x.groupby("behavior_family"):
        gg = g[g[c].notna()]; r[fam[:2]] = roc_auc_score(gg.ev, gg[c]) if gg.ev.nunique() == 2 else np.nan
    res.append(r)
print(pd.DataFrame(res).round(3).to_string())
t.to_parquet(f"{OUT}/r4/u12_artifacts.parquet")
