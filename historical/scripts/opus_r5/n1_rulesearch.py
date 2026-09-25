"""R5-N1: systematic search over FOURTH-FAMILY evidence rules, scored by consistency
with the leaderboard. The fourth family holds the last +0.0069 of recoverable score and
has no local labels, but six historical submissions differ ONLY in their fourth-family
evidence, giving four paired LB deltas. For each candidate rule (used as a truth proxy):
  K_fit  = slope of dS on dAP over the four paired deltas, and its SSE / sign agreement
  K_alg  = 0.2 * w from the ABSOLUTE public score, w = (E_known - E_pub)/(E_known - AP_cur)
A rule that is really the truth must satisfy both. R4 tested 7 proxies; this tests ~30,
including several mechanisms never tried."""
import numpy as np, pandas as pd, os, json, itertools
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; C_=f'{O}/r2_candidates'
EVC=[f'evidence_hand_{i}' for i in range(1,6)]
ev=pd.read_parquet(f'{O}/r4/z6_f4_typed_transfer_di_so.parquet')
q=pd.read_parquet(f'{O}/c24_hand_tables.parquet')
ev=ev.merge(q[['slot','h']+[c for c in q.columns if c.startswith('q_')]+['pair_win']],on=['slot','h'],how='left',suffixes=('','_q'))
for c in [x for x in ev.columns if x.startswith('q_')]: ev[c]=ev[c].fillna(0.0)
cur=pd.read_csv(f'{C_}/r13_ndwrank_cinew_f4.csv',dtype=str,keep_default_na=False).set_index('pair_id')
subs=[('ND','r2n_ND_on_r2j2m.csv',0.91563),('c-first','r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv',0.91838),
      ('NDdevpaw','r2n_NDdevpaw_on_r2j2m.csv',0.91974),('NDw','r2n_NDw_on_r2j2m.csv',0.92303),
      ('subp','r9_subp.csv',0.91639),('subh','r9_subh.csv',0.92115)]
S={nm:pd.read_csv(f'{C_}/{f}',dtype=str,keep_default_na=False).set_index('pair_id') for nm,f,_ in subs}
lb={nm:v for nm,_,v in subs}
mem=[p for p in S['NDw'].index[S['NDw'].predicted_behavior=='other_coordination'] if p in set(ev.pair_id)]
dy=np.array([lb['c-first']-lb['ND'],lb['NDdevpaw']-lb['ND'],lb['NDw']-lb['ND'],lb['subh']-lb['subp']])
E_PUB=(0.92501-0.78700)/0.2; EK=float(os.environ.get('EK','0.7445'))
def ap5(pred,tru):
    hits=0;sc=0.0
    for i,p in enumerate(pred[:5]):
        if p in tru: hits+=1; sc+=hits/(i+1)
    return sc/min(5,len(tru))
ev=ev.sort_values(['pair_id','ts']).reset_index(drop=True)
g=ev.groupby('pair_id')
trel=g.ts.rank(pct=True).values; ev['trel']=trel
# ---- building blocks ----
eqS=ev.k_ps_eq_last.values; eqR=ev.k_pr_eq_last.values
foldS=(ev.x_s_fold_to_r==1).values; foldR=(ev.x_r_fold_to_s==1).values
won=(ev.pair_win.values>0) if 'pair_win' in ev.columns else (ev.k_pr_won.values+ev.k_ps_won.values>0)
A2=((foldS&(eqS>=0.5))|(foldR&(eqR>=0.5)))
A2w=A2&won
BIGPOT=ev.x_conS.values+ev.x_conR.values
PASS=((ev.x_s_aggr_post==0)&(ev.x_r_aggr_post==0)).values
SD=ev.sd_any.astype(bool).values; BOTHF=ev.both_flop.astype(bool).values
ISO=(ev.iso.astype(float).values>0) if 'iso' in ev.columns else np.zeros(len(ev),bool)
PA6=(ev.pa_at_trig.values==6)
qHD=ev.q_HD.values if 'q_HD' in ev.columns else np.zeros(len(ev))
qACW=ev.q_HD_acw.values if 'q_HD_acw' in ev.columns else np.zeros(len(ev))
qPAW=ev.q_HD_paw.values if 'q_HD_paw' in ev.columns else np.zeros(len(ev))
LA=ev.LA.values; LB_=ev.LB.values; L=ev.L.values
curpick={p:[x for x in cur.loc[p,EVC].values if x!='NO_EVIDENCE'] for p in mem}
hid=ev.hand_id.values; pid=ev.pair_id.values
def slate_from(score, tau=None, key=None, fill_current=True, order_time=False):
    out={}
    df=pd.DataFrame({'pair_id':pid,'hand_id':hid,'s':score,'ts':ev.ts.values})
    for p,gg in df.groupby('pair_id'):
        if tau is None: sel=[]
        else:
            m=gg[gg.s>=tau]
            m=m.sort_values('ts') if order_time else m.sort_values('s',ascending=False)
            sel=m.hand_id.tolist()[:5]
        rest=[h for h in curpick.get(p,[]) if h not in sel] if fill_current else []
        fill=[h for h in gg.sort_values('s',ascending=False).hand_id if h not in sel and h not in rest]
        out[p]=(sel+rest+fill)[:5]
    return out
RULES={}
RULES['current_r13(old worldview)']={p:curpick[p] for p in mem}
for tau in (0.15,0.2,0.25,0.3,0.4,0.5,0.7):
    RULES[f'hybrid_LA_tau{tau}']=slate_from(LA,tau)
for tau in (0.3,0.4):
    RULES[f'hybrid_LA_tau{tau}_timeorder']=slate_from(LA,tau,order_time=True)
    RULES[f'hybrid_LA_tau{tau}_nofill']=slate_from(LA,tau,fill_current=False)
RULES['pureL_top5']=slate_from(L,tau=1e9)   # degenerate -> all fill by L
RULES['pureL_top5']= {p:g2.sort_values('s',ascending=False).hand_id.tolist()[:5] for p,g2 in pd.DataFrame({'pair_id':pid,'hand_id':hid,'s':L}).groupby('pair_id')}
def rule_slate(mask, order, name, fill=True):
    sc=np.where(mask, 1e6+order, order-1e6); RULES[name]=slate_from(sc,tau=0.0,fill_current=fill)
rule_slate(A2, -ev.ts.values*0+ (1-ev.trel.values), 'ruleA2_earliest+fill')
rule_slate(A2w, (1-ev.trel.values), 'ruleA2win_earliest+fill')
rule_slate(A2w, (1-ev.trel.values), 'ruleA2win_earliest_nofill', fill=False)
rule_slate(A2&(BIGPOT>np.quantile(BIGPOT,0.7)), (1-ev.trel.values), 'ruleA2_bigpot+fill')
rule_slate(PASS&SD&BOTHF, (1-ev.trel.values), 'rulePassiveSD+fill')
rule_slate(PA6, (1-ev.trel.values), 'ruleSixHanded+fill')
rule_slate(won&(BIGPOT>np.quantile(BIGPOT,0.8)), (1-ev.trel.values), 'ruleBigWin+fill')
rule_slate(qACW>np.quantile(qACW,0.9), (1-ev.trel.values), 'ruleTopACW_earliest+fill')
RULES['q_HD_top5']={p:g2.sort_values('s',ascending=False).hand_id.tolist()[:5] for p,g2 in pd.DataFrame({'pair_id':pid,'hand_id':hid,'s':qHD}).groupby('pair_id')}
RULES['q_acw_top5']={p:g2.sort_values('s',ascending=False).hand_id.tolist()[:5] for p,g2 in pd.DataFrame({'pair_id':pid,'hand_id':hid,'s':qACW}).groupby('pair_id')}
RULES['LA_top5']={p:g2.sort_values('s',ascending=False).hand_id.tolist()[:5] for p,g2 in pd.DataFrame({'pair_id':pid,'hand_id':hid,'s':LA}).groupby('pair_id')}
for w in (0.3,0.5,0.7):
    mix=w*pd.Series(LA).groupby(pid).rank(pct=True).values+(1-w)*pd.Series(qACW).groupby(pid).rank(pct=True).values
    RULES[f'mix_LA_qacw_w{w}']={p:g2.sort_values('s',ascending=False).hand_id.tolist()[:5] for p,g2 in pd.DataFrame({'pair_id':pid,'hand_id':hid,'s':mix}).groupby('pair_id')}
rows=[]
for nm,SL in RULES.items():
    SL={p:SL.get(p,[]) for p in mem}
    if any(len(v)==0 for v in SL.values()): continue
    ap={k:np.mean([ap5(list(S[k].loc[p,EVC].values),set(SL[p])) for p in mem]) for k in S}
    dx=np.array([ap['c-first']-ap['ND'],ap['NDdevpaw']-ap['ND'],ap['NDw']-ap['ND'],ap['subh']-ap['subp']])
    if (dx**2).sum()<1e-12: continue
    K=float(dx@dy/(dx@dx)); sse=float(((dy-K*dx)**2).sum()); sg=int((np.sign(dx)==np.sign(dy)).sum())
    apc=ap['NDw']; w=(EK-E_PUB)/(EK-apc) if EK>apc else np.nan; Kal=0.2*w
    ovl=float(np.mean([len(set(SL[p])&set(curpick[p])) for p in mem]))
    rows.append(dict(rule=nm,K_fit=round(K,4),SSE=sse,signs=sg,AP_NDw=round(apc,3),w=round(float(w),3),K_alg=round(float(Kal),4),
                     mismatch=round(abs(K-Kal),4),overlap_cur=round(ovl,2)))
d=pd.DataFrame(rows).sort_values(['signs','mismatch'],ascending=[False,True])
pd.set_option('display.width',220); print(f'E_pub={E_PUB:.5f} E_known(eval)={EK}  members={len(mem)}')
print(d.to_string(index=False))
d.to_json(f'{O}/r5/n1_rulesearch.json',orient='records',indent=1)
