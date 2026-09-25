"""R5-F1: pick the fourth-family hybrid threshold by a criterion R4 did not have.

R4 fitted one constant K in  dS_j = K * dAP_j(proxy)  over four paired LB deltas
and accepted any K inside a loose 'theoretical' band 0.026-0.033.
A second, independent equation pins K from the ABSOLUTE public score:
    E_pub = (1-w) E_known(eval) + w AP_F4(current slate),  K = 0.2 w
with E_known(eval) measured by re-weighting dev per-pair AP to the eval exposure
(R5-d4: 0.7456 for the r10_ci configuration) and AP_F4(current) read off the proxy.
A proxy that is really the truth must satisfy BOTH. Report K_fit vs K_algebra."""
import numpy as np, pandas as pd, os, json
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; C_=f'{O}/r2_candidates'
EVC=[f'evidence_hand_{i}' for i in range(1,6)]
ev=pd.read_parquet(f'{O}/r4/z6_f4_typed_transfer_di_so.parquet')
cur=pd.read_csv(f'{C_}/r13_ndwrank_cinew_f4.csv',dtype=str,keep_default_na=False).set_index('pair_id')
subs=[('ND','r2n_ND_on_r2j2m.csv',0.91563),('c-first','r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv',0.91838),('NDdevpaw','r2n_NDdevpaw_on_r2j2m.csv',0.91974),('NDw','r2n_NDw_on_r2j2m.csv',0.92303),('subp','r9_subp.csv',0.91639),('subh','r9_subh.csv',0.92115)]
S={nm:pd.read_csv(f'{C_}/{f}',dtype=str,keep_default_na=False).set_index('pair_id') for nm,f,_ in subs}; lb={nm:v for nm,_,v in subs}
mem=[p for p in S['NDw'].index[S['NDw'].predicted_behavior=='other_coordination'] if p in set(ev.pair_id)]
dy=np.array([lb['c-first']-lb['ND'],lb['NDdevpaw']-lb['ND'],lb['NDw']-lb['ND'],lb['subh']-lb['subp']])
E_PUB=(0.92501-0.78700)/0.2                    # r10_ci public evidence score
EK=float(os.environ.get('EK','0.7456'))        # exposure-matched known-family E of the r10_ci configuration
def ap5(pred,tru):
    hits=0;sc=0.0
    for i,p in enumerate(pred[:5]):
        if p in tru: hits+=1; sc+=hits/(i+1)
    return sc/min(5,len(tru))
def slate(g,pid,tau):
    a=g[g.LA>=tau].sort_values('LA',ascending=False).hand_id.tolist()[:5]
    rest=[h for h in cur.loc[pid,EVC].values if h not in a and h!='NO_EVIDENCE']
    fill=[h for h in g.sort_values('L',ascending=False).hand_id.tolist() if h not in a and h not in rest]
    return (a+rest+fill)[:5],len(a)
rows=[]
for tau in (0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.85,1.2,2.0):
    SL={};na=[]
    for pid,g in ev.groupby('pair_id'):
        SL[pid],k=slate(g,pid,tau); na.append(k)
    ap={nm:np.mean([ap5(list(S[nm].loc[p,EVC].values),set(SL[p])) for p in mem]) for nm in S}
    dx=np.array([ap['c-first']-ap['ND'],ap['NDdevpaw']-ap['ND'],ap['NDw']-ap['ND'],ap['subh']-ap['subp']])
    K=float(dx@dy/(dx@dx)); sse=float(((dy-K*dx)**2).sum())
    apNDw=ap['NDw']                              # r10_ci carries the NDw fourth-family slate
    w=(EK-E_PUB)/(EK-apNDw) if EK>apNDw else np.nan
    rows.append(dict(tau=tau,nA=round(float(np.mean(na)),2),
        ov=round(float(np.mean([len(set(SL[p])&set(cur.loc[p,EVC].values)) for p in SL])),2),
        K_fit=round(K,4),SSE=sse,signs=int((np.sign(dx)==np.sign(dy)).sum()),
        AP_NDw=round(apNDw,3),w_alg=round(float(w),3),K_alg=round(0.2*float(w),4),
        mismatch=round(abs(K-0.2*float(w)),4)))
d=pd.DataFrame(rows); pd.set_option('display.width',220); print(f'E_pub={E_PUB:.5f}  E_known(eval)={EK}')
print(d.to_string(index=False))
print()
print('upside if the fourth family were lifted to the known-family level, at each tau:')
for _,r in d.iterrows():
    print(f"  tau {r.tau}: w={r.w_alg}  dS(F4 -> {EK}) = {0.2*r.w_alg*(EK-r.AP_NDw):+.4f}")
