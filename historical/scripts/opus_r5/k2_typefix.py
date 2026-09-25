"""R5-K2: the two-type decoder's TRAINING LABELS have never been questioned.

The evidence list is [A events by time] + [B events by time]. A pair whose listed
hands form ONE ascending run therefore has evidence that is ENTIRELY one type - a
single-run pair cannot be a mix. But g4/g8/z6 type every hand independently with a
classifier, so 38% of DT pairs (57/148) and 27% of SP pairs can be handed a
structurally impossible mixed label, corrupting p_A and p_B and the censoring masks.

Variants (same protocol as R4-G4: pool GroupKFold(5) x 3 seeds, official AP@5 inside
the frozen R15 top-20, full-truth denominators):
  base     per-hand typing (current)
  pairfix  one type per single-run pair (mean classifier probability over its hands)
  xfam     type classifier trained on BOTH DT and SP two-run pairs (type A is the same
           event in both, so this doubles the typing data)
  soft     no hard type: train p_A with sample weight q_A and p_B with 1-q_A
  pairfix+xfam, pairfix+xfam+soft
"""
import numpy as np, pandas as pd, lightgbm as lgb, sys, os, json
sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/r18'); sys.path.insert(0,'/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920')
from sklearn.model_selection import GroupKFold
import ci_censored_event as C
from g4_decode import decode, runs
O='/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917'; R5=f'{O}/r5'; NJ=int(os.environ.get('NJ',6))
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9)))
SEEDS=(260919,11,29)
def assemble():
    base=C.prepare(pd.read_parquet(f'{O}/t5_dev_seq.parquet'))
    nf=pd.read_parquet(f'{O}/r3/t58_seq_feats.parquet'); NEW=[c for c in nf.columns if c not in ('slot','h','pa','pb')]
    base=base.merge(nf[['slot','h']+NEW],on=['slot','h'],how='left'); base[NEW]=base[NEW].fillna(0.0)
    role=pd.read_parquet(f'{O}/r4/x2c_role_dev.parquet'); ROLE=[c for c in role.columns if c.startswith('x_') and c not in ('x_k','x_n')]
    ker=pd.read_parquet(f'{O}/r4/x11_kernel_dev.parquet'); KER=[c for c in ker.columns if c.startswith('k_')]
    d=base.merge(role[['slot','h']+ROLE],on=['slot','h'],validate='one_to_one').merge(ker[['slot','h']+KER],on=['slot','h'],validate='one_to_one')
    return d, C.FEATURES+NEW+ROLE+KER
full,FS=assemble()
def symmetrise(d):
    out=[];cols=set(d.columns); new={}
    pairs=[(c,c.replace('k_rs_','k_sr_')) for c in cols if c.startswith('k_rs_')]+[(c,c.replace('k_pr_','k_ps_')) for c in cols if c.startswith('k_pr_')]
    pairs+=[('x_netR','x_netS'),('x_conR','x_conS'),('x_r_aggr_pre','x_s_aggr_pre'),('x_r_aggr_post','x_s_aggr_post'),('x_r_last','x_s_last'),('x_r_last_st','x_s_last_st'),('x_sdR','x_sdS'),('x_hsR_pre','x_hsS_pre'),('x_hsR_last','x_hsS_last'),('x_r_fold_to_s','x_s_fold_to_r'),('x_r_aggr_n','x_s_aggr_n')]
    for a,b in pairs:
        if a in cols and b in cols:
            nm=a.replace('k_rs_','y_i_').replace('k_pr_','y_p_').replace('x_','y_x_'); new[nm+'_mx']=np.maximum(d[a],d[b]); new[nm+'_mn']=np.minimum(d[a],d[b]); out+=[nm+'_mx',nm+'_mn']
    sf=d.x_s_fold_to_r==1; rf=d.x_r_fold_to_s==1
    new['y_folder_eq']=np.where(sf,d.k_ps_eq_last,np.where(rf,d.k_pr_eq_last,-1.0)); new['y_bettor_eq']=np.where(sf,d.k_pr_eq_last,np.where(rf,d.k_ps_eq_last,-1.0))
    new['y_folder_hs']=np.where(sf,d.x_hsS_last,np.where(rf,d.x_hsR_last,-1.0)); new['y_folder_contrib']=np.where(sf,d.x_conS,np.where(rf,d.x_conR,-1.0)); new['y_eq_gap_abs']=(d.k_ps_eq_last-d.k_pr_eq_last).abs()
    return pd.concat([d,pd.DataFrame(new,index=d.index)],axis=1), out+['y_folder_eq','y_bettor_eq','y_folder_hs','y_folder_contrib','y_eq_gap_abs']
full,SY=symmetrise(full); FSA=FS+SY
g1=pd.read_parquet(f'{O}/r4/g1_evidence_rank.parquet')[['h','pair_id','evidence_rank','chron']].sort_values(['pair_id','evidence_rank'])
g1['run']=np.concatenate([runs(g.chron.values) for _,g in g1.groupby('pair_id',sort=False)]); g1['nruns']=g1.groupby('pair_id').run.transform('max')+1
full=full.merge(g1[['h','evidence_rank','run','nruns']],on='h',how='left').sort_values(['slot','ts','h'],kind='stable').reset_index(drop=True)
cand=pd.read_parquet(f'{O}/t4_wrong_vs_hit.parquet').rename(columns={'sl':'slot'})
hmap=pd.read_parquet(f'{O}/np/hand_index.parquet').set_index('hand_id').hi
t45=pd.read_parquet(f'{O}/r3/t45_known_e_rerank.parquet'); t45['h']=t45.hand_id.map(hmap)
pools=np.array(sorted(full.pool.unique()))
# ---- type classifiers ----
def fit_type(mask_fams):
    e=full[full.fam.isin(mask_fams)&full.ev.astype(bool)]; two=e[e.nruns==2]
    return lgb.LGBMClassifier(n_estimators=200,learning_rate=.05,num_leaves=7,min_child_samples=10,colsample_bytree=.5,verbosity=-1,n_jobs=NJ).fit(two[FSA].astype(float),(two.run==1).astype(int))
CLF={}
res={}
for FAM in ('directed_transfer','soft_play'):
    s=full[full.fam==FAM].reset_index(drop=True); counts=s.groupby('slot').ev.sum()
    c=cand[cand.slot.isin(s.slot)].drop(columns=['ev','ts'],errors='ignore').merge(s[['slot','h','ts','ev']],on=['slot','h'],validate='one_to_one').merge(t45[['slot','h','tab']],on=['slot','h'],how='left').reset_index(drop=True)
    fmi=pd.MultiIndex.from_arrays([s.slot,s.h]); mi=pd.MultiIndex.from_arrays([c.slot,c.h]); to_c=lambda v: pd.Series(v,index=fmi).reindex(mi).values
    TAB=lg(c.tab.values); R15=-c.r.values.astype(float)
    for key,fams in (('own',[FAM]),('xfam',['directed_transfer','soft_play'])):
        if key not in CLF: CLF[key]={}
        CLF[key][FAM]=fit_type(fams)
    out={}
    for VAR in ('base','pairfix','xfam','pairfix_xfam','pairfix_xfam_soft'):
        clf=CLF['xfam' if 'xfam' in VAR else 'own'][FAM]
        qB=clf.predict_proba(s[FSA].astype(float))[:,1]                     # P(this hand is type B)
        ev=s.ev.astype(bool).values
        if 'pairfix' in VAR:   # a single-run pair's evidence is ENTIRELY one type
            df=pd.DataFrame({'slot':s.slot.values,'qB':qB,'ev':ev,'nruns':s.nruns.values})
            m1=df.ev&(df.nruns==1)
            pm=df[m1].groupby('slot').qB.mean()
            qB=np.where(m1.values, df.slot.map(pm).fillna(0.5).values, qB)
        typB=np.where(s.nruns.values==2, s.run.values==1, qB>0.5)
        if VAR=='base': nmix=int(pd.DataFrame({'slot':s.slot.values,'t':typB,'ev':ev})[ev].groupby('slot').t.nunique().gt(1).sum())
        s2=s.copy(); s2['isA']=ev&~typB; s2['isB']=ev&typB
        nA=s2.groupby('slot').isA.transform('sum'); nB=s2.groupby('slot').isB.transform('sum'); nev=s2.groupby('slot').ev.transform('sum')
        lastA=s2.slot.map(s2[s2.isA].groupby('slot').ts.max()); lastB=s2.slot.map(s2[s2.isB].groupby('slot').ts.max())
        incA=((nA<5)|(s2.ts<=lastA)).values; incB=((nev<5)|((nB>0)&(s2.ts<=lastB))).values
        wA=np.where(ev, np.where(s.nruns.values==2,(s.run.values==0).astype(float),1-qB), 1.0)
        wB=np.where(ev, np.where(s.nruns.values==2,(s.run.values==1).astype(float),qB), 1.0)
        aps={'alone':[],'stack1':[],'rank035':[]}
        for seed in SEEDS:
            pA=np.zeros(len(s)); pB=np.zeros(len(s))
            for _,vi in GroupKFold(5,shuffle=True,random_state=seed).split(pools,groups=pools):
                vp=pools[vi]; tr=(~s.pool.isin(vp)).to_numpy(); va=~tr
                if 'soft' in VAR:
                    mA=lgb.LGBMClassifier(**{**C.PARAMS,'n_jobs':NJ,'random_state':seed}).fit(s.loc[tr&incA,FSA].astype(float),(ev&(wA>0))[tr&incA].astype(int),sample_weight=np.where(ev[tr&incA],np.maximum(wA[tr&incA],1e-3),1.0))
                    mB=lgb.LGBMClassifier(**{**C.PARAMS,'n_jobs':NJ,'random_state':seed}).fit(s.loc[tr&incB,FSA].astype(float),(ev&(wB>0))[tr&incB].astype(int),sample_weight=np.where(ev[tr&incB],np.maximum(wB[tr&incB],1e-3),1.0))
                else:
                    mA=lgb.LGBMClassifier(**{**C.PARAMS,'n_jobs':NJ,'random_state':seed}).fit(s.loc[tr&incA,FSA].astype(float),s2.loc[tr&incA,'isA'].astype(int))
                    mB=lgb.LGBMClassifier(**{**C.PARAMS,'n_jobs':NJ,'random_state':seed}).fit(s.loc[tr&incB,FSA].astype(float),s2.loc[tr&incB,'isB'].astype(int))
                pA[va]=mA.predict_proba(s.loc[va,FSA].astype(float))[:,1]; pB[va]=mB.predict_proba(s.loc[va,FSA].astype(float))[:,1]
            L,_,_=decode(s,np.clip(pA,1e-9,1-1e-6),np.clip(pB,1e-9,1-1e-6)); v=to_c(L)
            aps['alone'].append(C.pair_ap(c,v,counts)); aps['stack1'].append(C.pair_ap(c,TAB+lg(v),counts))
            rk=lambda x: pd.Series(x).groupby(c.slot.values).rank(pct=True).values
            aps['rank035'].append(C.pair_ap(c,0.65*rk(R15)+0.35*rk(v),counts))
        for k,v2 in aps.items(): out[f'{VAR}|{k}']=round(float(np.mean([a.mean() for a in v2])),4)
        out.setdefault('_per',{})[VAR]=aps
        print(f'  {FAM[:2]} {VAR:20s} ' + ' '.join(f'{k} {out[f"{VAR}|{k}"]:.4f}' for k in ('alone','stack1','rank035')),flush=True)
    per=out.pop('_per')
    ref='base'
    rng=np.random.default_rng(7); idx=np.array(counts.index); pl=idx//900
    form={'directed_transfer':'stack1','soft_play':'rank035'}[FAM]
    for VAR in per:
        if VAR==ref: continue
        d=np.mean([per[VAR][form][i].reindex(idx).values-per[ref][form][i].reindex(idx).values for i in range(len(SEEDS))],axis=0)
        pm=pd.Series(d,index=pl).groupby(level=0).mean(); bs=np.array([pm.values[rng.integers(0,len(pm),len(pm))].mean() for _ in range(4000)])
        print(f'   {FAM[:2]} {VAR:20s} vs base [{form}] d={d.mean():+.4f} poolP={float((bs>0).mean()):.3f}',flush=True)
    out['n_impossible_mixed_single_run_pairs_base']=nmix; res[FAM]=out
json.dump(res,open(f'{R5}/k2_typefix.json','w'),indent=1)
