from common import *
df = pd.read_parquet(f"{OUT}/lab_pairhand_dev_v1.parquet")
df["t_max"] = df[["t_ab","t_ba"]].max(axis=1)
df["t_abs"] = (df.a_net_bb - df.b_net_bb).abs()
df["pot_bb"] = df.final_pot / df.bb
df["both_vpip"] = (df.a_vpip & df.b_vpip).astype(int)
df["ab_contrib"] = df.a_c_bb + df.b_c_bb
df["ab_min_contrib"] = df[["a_c_bb","b_c_bb"]].min(axis=1)
df["fold_between"] = df.a_fold_to_b + df.b_fold_to_a
df["tsec"] = pd.to_datetime(df.started_at).astype("int64")//10**9
pos = df[df.label==1].copy()
crit = ["t_max","t_abs","pot_bb","ab_contrib","ab_min_contrib","o_fold_to_ab","fold_between","ab_hu_checks","players_at_showdown"]
for fam in ["directed_transfer","soft_play","coordinated_isolation"]:
    sub = pos[pos.fam==fam]
    print("=====", fam, "pairs", sub.pair_id.nunique(), "hands", len(sub))
    for c in crit:
        r = sub.groupby("pair_id")[c].rank(ascending=False, method="min")
        evr = r[sub.is_ev]
        print(f"{c:22s} ev within-pair rank: med={evr.median():.0f} p90={evr.quantile(.9):.0f} frac<=5={np.mean(evr<=5):.3f} frac<=10={np.mean(evr<=10):.3f}")
    # time: evidence hands position in pair's timeline
    rt = sub.groupby("pair_id")["tsec"].rank(pct=True)
    print("time pct of ev hands: quartiles", np.round(np.quantile(rt[sub.is_ev],[.1,.25,.5,.75,.9]),3), " non:", np.round(np.quantile(rt[~sub.is_ev],[.25,.5,.75]),3))
    # evidence_rank vs time order and vs t_max
    e = sub[sub.is_ev].copy()
    e["trank"] = e.groupby("pair_id")["tsec"].rank()
    e["tmaxrank"] = e.groupby("pair_id")["t_max"].rank(ascending=False)
    e["potrank"] = e.groupby("pair_id")["pot_bb"].rank(ascending=False)
    print("corr(evidence_rank, time rank)=%.3f corr(evidence_rank, tmax rank)=%.3f corr(ev_rank,pot rank)=%.3f" % (e[["evidence_rank","trank"]].corr().iloc[0,1], e[["evidence_rank","tmaxrank"]].corr().iloc[0,1], e[["evidence_rank","potrank"]].corr().iloc[0,1]))
    print(pd.crosstab(e.evidence_rank, e.trank))
# how many "evidence-like" hands per positive pair vs negative pairs, for a simple family signature
def sig(d, fam):
    if fam=="directed_transfer":
        return (d.both_vpip==1) & (d.t_max>=5) & ((d.a_fold_to_o + d.b_fold_to_o)==0)
    if fam=="soft_play":
        return (d.both_vpip==1) & (d.ab_hu_checks>=1)
    return (d.o_fold_to_ab>=2) & (d.a_aggr>=1) & (d.b_aggr>=1)
for fam in ["directed_transfer","soft_play","coordinated_isolation"]:
    s_ev = sig(pos[(pos.fam==fam)&pos.is_ev], fam).mean()
    s_non = sig(pos[(pos.fam==fam)&~pos.is_ev], fam).mean()
    s_neg = sig(df[df.label==0], fam).mean()
    cnt_pos = pos[pos.fam==fam].assign(s=lambda d: sig(d,fam)).groupby("pair_id").s.sum()
    cnt_neg = df[df.label==0].assign(s=lambda d: sig(d,fam)).groupby("pair_id").s.sum()
    print(f"{fam}: sig rate ev={s_ev:.3f} non-ev(pos)={s_non:.3f} neg={s_neg:.3f}; per-pair count pos q={np.quantile(cnt_pos,[.1,.5,.9])} neg q={np.quantile(cnt_neg,[.5,.9,.99])}")
# evidence count k vs shared hands
k = pos[pos.is_ev].groupby("pair_id").size().rename("k")
n = pos.groupby("pair_id").size().rename("n")
kk = pd.concat([k,n],axis=1).join(pos.groupby("pair_id").fam.first())
print(kk.groupby(["fam","k"]).n.describe())
