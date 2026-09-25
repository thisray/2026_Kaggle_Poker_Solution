"""R5-N2: how much of the fourth-family truth does the tau=0.3 hybrid explain?
Model the truth as a MIXTURE: with probability alpha a pair's evidence follows the
hybrid proxy, otherwise the old (M3-ACT) proxy. Then
   AP_j = alpha*AP_j(hybrid) + (1-alpha)*AP_j(current)   and   dS_j = K * dAP_j
Fit (K, alpha) on the four paired LB deltas, and check against the third, independent
equation K = 0.2*w from the absolute public score. alpha near 1 means bet fully on the
hybrid; alpha near 0.5 means the optimal slate should hedge."""
import numpy as np, pandas as pd, os, json
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; C_=f'{O}/r2_candidates'
EVC=[f'evidence_hand_{i}' for i in range(1,6)]
ev=pd.read_parquet(f'{O}/r4/z6_f4_typed_transfer_di_so.parquet').sort_values(['pair_id','ts']).reset_index(drop=True)
cur=pd.read_csv(f'{C_}/r13_ndwrank_cinew_f4.csv',dtype=str,keep_default_na=False).set_index('pair_id')
subs=[('ND','r2n_ND_on_r2j2m.csv',0.91563),('c-first','r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv',0.91838),
      ('NDdevpaw','r2n_NDdevpaw_on_r2j2m.csv',0.91974),('NDw','r2n_NDw_on_r2j2m.csv',0.92303),
      ('subp','r9_subp.csv',0.91639),('subh','r9_subh.csv',0.92115)]
S={nm:pd.read_csv(f'{C_}/{f}',dtype=str,keep_default_na=False).set_index('pair_id') for nm,f,_ in subs}
lb={nm:v for nm,_,v in subs}
mem=[p for p in S['NDw'].index[S['NDw'].predicted_behavior=='other_coordination'] if p in set(ev.pair_id)]
dy=np.array([lb['c-first']-lb['ND'],lb['NDdevpaw']-lb['ND'],lb['NDw']-lb['ND'],lb['subh']-lb['subp']])
E_PUB=(0.92501-0.78700)/0.2
def ap5(pred,tru):
    hits=0;sc=0.0
    for i,p in enumerate(pred[:5]):
        if p in tru: hits+=1; sc+=hits/(i+1)
    return sc/min(5,len(tru))
curpick={p:[x for x in cur.loc[p,EVC].values if x!='NO_EVIDENCE'] for p in mem}
def hybrid(tau):
    out={}
    for p,g in ev.groupby('pair_id'):
        a=g[g.LA>=tau].sort_values('LA',ascending=False).hand_id.tolist()[:5]
        rest=[h for h in curpick.get(p,[]) if h not in a]
        fill=[h for h in g.sort_values('L',ascending=False).hand_id if h not in a and h not in rest]
        out[p]=(a+rest+fill)[:5]
    return out
H=hybrid(0.3)
apH={k:np.array([ap5(list(S[k].loc[p,EVC].values),set(H[p])) for p in mem]) for k in S}
apC={k:np.array([ap5(list(S[k].loc[p,EVC].values),set(curpick[p])) for p in mem]) for k in S}
def deltas(ap):
    m={k:ap[k].mean() for k in ap}
    return np.array([m['c-first']-m['ND'],m['NDdevpaw']-m['ND'],m['NDw']-m['ND'],m['subh']-m['subp']])
dH,dC=deltas(apH),deltas(apC)
print(f'dy (LB)        = {np.round(dy,5)}')
print(f'dAP hybrid     = {np.round(dH,4)}')
print(f'dAP current    = {np.round(dC,4)}')
best=None
print(f'{"alpha":>6s} {"K_fit":>8s} {"SSE":>10s} {"AP_NDw":>7s} {"w":>6s} {"K_alg":>7s} {"mismatch":>9s}')
for a in np.arange(0.0,1.0001,0.05):
    dx=a*dH+(1-a)*dC
    if (dx**2).sum()<1e-14: continue
    K=float(dx@dy/(dx@dx)); sse=float(((dy-K*dx)**2).sum())
    apc=a*apH['NDw'].mean()+(1-a)*apC['NDw'].mean()
    for EK in (0.7445,):
        w=(EK-E_PUB)/(EK-apc) if EK>apc else np.nan; Kal=0.2*w; mm=abs(K-Kal)
    if a in (0.0,0.2,0.4,0.5,0.6,0.7,0.8,0.9,1.0) or (best and mm<best[0]):
        print(f'{a:6.2f} {K:8.4f} {sse:10.2e} {apc:7.3f} {w:6.3f} {Kal:7.4f} {mm:9.4f}')
    if best is None or mm<best[0]: best=(mm,a,K,sse,apc)
mm,a,K,sse,apc=best
print(f'\nbest joint solution: alpha={a:.2f}  K={K:.4f}  SSE={sse:.2e}  AP(current slate|mixture truth)={apc:.3f}  |K_fit-K_alg|={mm:.4f}')
print(f'  -> the fourth family is ~{a*100:.0f}% explained by the two-type hybrid structure')
json.dump(dict(alpha=float(a),K=float(K),SSE=float(sse),AP_cur=float(apc),mismatch=float(mm)),open(f'{O}/r5/n2_mixture.json','w'),indent=1)
