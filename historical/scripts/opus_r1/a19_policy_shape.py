import numpy as np, pandas as pd
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
X = np.load(f"{OUT}/dec_X.npy", mmap_mode="r"); Y = np.load(f"{OUT}/dec_Y.npy", mmap_mode="r")
rng = np.random.RandomState(0); idx = np.sort(rng.choice(len(Y), 3_000_000, replace=False))
FN = ["st","facing","tc_bb","pot_bb","pot_odds","stack_bb","spr","n_active","pos","is_blind","n_aggr_st","actor_aggr_st","initiative","contrib_st_bb","hs1","hs2","cat","pf_eq","sty_vpip","sty_pfr","sty_agg","sty_fold_facing","sty_call_facing","tilt10","tilt_last","hands_seen"]
d = pd.DataFrame(np.asarray(X[idx]), columns=FN); d["y"] = np.asarray(Y[idx])
# postflop, facing a bet, heads-up (n_active==2): fold/call/raise frequencies by hs1 decile and pot odds bin
g = d[(d.st >= 1) & (d.facing == 1) & (d.n_active == 2)].copy()
g["hsb"] = pd.cut(g.hs1, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]); g["pob"] = pd.cut(g.pot_odds, [0, .2, .3, .4, .5, 1])
t = g.groupby(["pob", "hsb"], observed=True).y.apply(lambda s: pd.Series({"fold": (s == 0).mean(), "call": (s == 2).mean(), "raise": (s == 3).mean(), "n": len(s)})).unstack()
print("postflop HU facing bet: action freq by pot-odds bin x hs1 bin"); print(t.round(3).to_string())
# not facing: bet frequency by hs1
g2 = d[(d.st >= 1) & (d.facing == 0) & (d.n_active == 2)].copy(); g2["hsb"] = pd.cut(g2.hs1, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
print("postflop HU not facing: bet freq by hs1"); print(g2.groupby("hsb", observed=True).y.apply(lambda s: pd.Series({"bet": (s == 3).mean(), "n": len(s)})).unstack().round(3).to_string())
# preflop open (not facing a raise beyond blind: n_aggr_st==0), raise freq by pf_eq
g3 = d[(d.st == 0) & (d.n_aggr_st == 0)].copy(); g3["pfb"] = pd.cut(g3.pf_eq, [0, .35, .4, .45, .5, .55, .6, .65, .7, .9])
print("preflop unopened: action freq by preflop equity"); print(g3.groupby("pfb", observed=True).y.apply(lambda s: pd.Series({"fold": (s == 0).mean(), "call": (s == 2).mean(), "raise": (s == 3).mean(), "n": len(s)})).unstack().round(3).to_string())
# within a narrow cell, does style explain residual? correlation of raise freq with sty_agg
cell = g3[(g3.pf_eq > 0.5) & (g3.pf_eq < 0.55)]
cell["aggb"] = pd.qcut(cell.sty_agg, 5)
print("preflop unopened pf_eq 0.50-0.55: raise freq by player aggression quintile"); print(cell.groupby("aggb", observed=True).y.apply(lambda s: (s == 3).mean()).round(3).to_string())
