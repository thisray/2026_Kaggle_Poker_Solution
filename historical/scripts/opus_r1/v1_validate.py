from common import *
D = f"{OUT}/np"
R = np.load(f"{OUT}/R_v1.npy", mmap_mode="r"); P = np.load(f"{OUT}/P_v1.npy", mmap_mode="r")
eqm = np.load(f"{OUT}/act_eqm_v1.npy", mmap_mode="r"); eqla = np.load(f"{OUT}/act_eqla_v1.npy", mmap_mode="r")
act = np.load(f"{D}/a_act.npy", mmap_mode="r"); st = np.load(f"{D}/a_st.npy", mmap_mode="r"); tc = np.load(f"{D}/a_to_call.npy", mmap_mode="r"); amt=np.load(f"{D}/a_amount.npy", mmap_mode="r")
pa = np.load(f"{D}/a_players_active.npy", mmap_mode="r")
names = open(f"{OUT}/feature_names_v1.txt").read().split("\n")
RN = names[0][2:].split(","); PN = names[1][2:].split(",")
# equity sanity by action type and street
df = pd.DataFrame({"act": act[:3000000], "st": st[:3000000], "eqm": eqm[:3000000], "eqla": eqla[:3000000], "pa": pa[:3000000]})
print(df.groupby(["st","act"]).agg(n=("eqm","size"), eqm=("eqm","mean"), eqla=("eqla", lambda x: x[x>=0].mean())).round(3).to_string())
print("HU river eqm distribution for fold/call/raise:")
d2 = df[(df.st==3)&(df.pa==2)]
print(d2.groupby("act").eqm.describe().round(3))
# compare with DuckDB features for labelled pairs
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); pidx = pd.read_parquet(f"{D}/player_index.parquet")
lab = pd.read_parquet(f"{OUT}/lab_pairhand_dev_v1.parquet")
labels = pd.read_csv(f"{RAW}/development_labels.csv")
lab = lab.merge(hidx, on="hand_id").merge(labels[["pair_id","player_1","player_2"]], on="pair_id")
pmap = dict(zip(pidx.player_id, pidx.pi))
lab["pa_i"] = lab.player_1.map(pmap); lab["pb_i"] = lab.player_2.map(pmap)
sp = np.load(f"{D}/s_player.npy", mmap_mode="r")
hi = lab.hi.values
seatA = np.argmax(sp[hi] == lab.pa_i.values[:,None], axis=1); seatB = np.argmax(sp[hi] == lab.pb_i.values[:,None], axis=1)
assert (sp[hi, seatA] == lab.pa_i.values).all() and (sp[hi, seatB] == lab.pb_i.values).all()
for nm, col in [("fold_to","a_fold_to_b"),("call_to","a_call_b"),("raise_over","a_reraise_b")]:
    k = RN.index(nm)
    v = R[hi, seatA, seatB, k]
    print(nm, "agree frac", np.mean(v == lab[col].values), "mean kernel", v.mean(), "mean duck", lab[col].mean())
vp = P[hi, seatA, PN.index("vpip")]
print("vpip agree", np.mean(vp == lab.a_vpip.astype(int).values))
cb = P[hi, seatA, PN.index("contrib_bb")]
print("contrib agree", np.mean(np.isclose(cb, lab.a_c_bb.values)))
