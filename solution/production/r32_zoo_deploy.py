"""R5-G6Z: deploy the PROBABILITY-LEVEL fusion of the 14-config event zoo to eval.
Same shape as R4's g6_deploy_typed.py, but p_A / p_B are the equal-weight logit mean
over every zoo config (and 3 seeds each) before a single pass of the two-type decoder.
Validated on dev by R5-e3: soft_play PFUSE_ALL + R15 rank blend w=0.35 = 0.7308 vs the
deployed R15 0.7198 (d=+0.0110, pool bootstrap P=0.971, pair bootstrap P=0.956).
usage: python g6z_deploy_zoo.py <family> <rank|stack> <PAR> <full> <new> <role> <ker> <cand> <out.csv>"""
import numpy as np, pandas as pd, lightgbm as lgb, json, sys, os, hashlib
import ci_censored_event as C
from g4_decode import decode, runs
O=os.environ['POKER_WORK_DIR']; NJ=int(os.environ.get('NJ',4))
fam,MODE,PAR=sys.argv[1],sys.argv[2],float(sys.argv[3]); f_full,f_new,f_role,f_ker,f_cand,out_csv=sys.argv[4:10]
def symmetrise(d):
    out=[];cols=set(d.columns); new={}
    pairs=[(c,c.replace('k_rs_','k_sr_')) for c in cols if c.startswith('k_rs_')]+[(c,c.replace('k_pr_','k_ps_')) for c in cols if c.startswith('k_pr_')]
    pairs+=[('x_netR','x_netS'),('x_conR','x_conS'),('x_r_aggr_pre','x_s_aggr_pre'),('x_r_aggr_post','x_s_aggr_post'),('x_r_last','x_s_last'),('x_r_last_st','x_s_last_st'),('x_sdR','x_sdS'),('x_hsR_pre','x_hsS_pre'),('x_hsR_last','x_hsS_last'),('x_r_fold_to_s','x_s_fold_to_r'),('x_r_aggr_n','x_s_aggr_n')]
    for a,b in pairs:
        if a in cols and b in cols:
            nm=a.replace('k_rs_','y_i_').replace('k_pr_','y_p_').replace('x_','y_x_'); new[nm+'_mx']=np.maximum(d[a],d[b]); new[nm+'_mn']=np.minimum(d[a],d[b]); out+=[nm+'_mx',nm+'_mn']
    sf=d.x_s_fold_to_r==1; rf=d.x_r_fold_to_s==1
    new['y_folder_eq']=np.where(sf,d.k_ps_eq_last,np.where(rf,d.k_pr_eq_last,-1.0)); new['y_bettor_eq']=np.where(sf,d.k_pr_eq_last,np.where(rf,d.k_ps_eq_last,-1.0)); new['y_folder_hs']=np.where(sf,d.x_hsS_last,np.where(rf,d.x_hsR_last,-1.0))
    new['y_folder_contrib']=np.where(sf,d.x_conS,np.where(rf,d.x_conR,-1.0)); new['y_eq_gap_abs']=(d.k_ps_eq_last-d.k_pr_eq_last).abs()
    d=pd.concat([d,pd.DataFrame(new,index=d.index)],axis=1)
    return d,out+['y_folder_eq','y_bettor_eq','y_folder_hs','y_folder_contrib','y_eq_gap_abs']
def assemble(frame,newf,role,ker):
    NEW=[c for c in newf.columns if c not in ('slot','h','pa','pb','pair_id')]; ROLE=[c for c in role.columns if c.startswith('x_') and c not in ('x_k','x_n')]; KER=[c for c in ker.columns if c.startswith('k_')]
    d=frame.merge(newf[['slot','h']+NEW],on=['slot','h'],how='left',validate='one_to_one'); d[NEW]=d[NEW].fillna(0.0)
    d=d.merge(role[['slot','h']+ROLE],on=['slot','h'],validate='one_to_one').merge(ker[['slot','h']+KER],on=['slot','h'],validate='one_to_one')
    assert len(d)==len(frame) and d[ROLE+KER].notna().all().all()
    d=d.sort_values(['slot','ts','h'],kind='stable').reset_index(drop=True); d,SY=symmetrise(d)
    return d, C.FEATURES+NEW+ROLE+KER, SY
dev_all=C.prepare(pd.read_parquet(f'{O}/t5_dev_seq.parquet')); dev_all=dev_all[dev_all.fam==fam].reset_index(drop=True)
s,FSb,SY=assemble(dev_all,pd.read_parquet(f'{O}/r3/t58_seq_feats.parquet'),pd.read_parquet(f'{O}/r4/x2c_role_dev.parquet'),pd.read_parquet(f'{O}/r4/x11_kernel_dev.parquet'))
ev,FSb2,SY2=assemble(C.prepare(pd.read_parquet(f_full)),pd.read_parquet(f_new),pd.read_parquet(f_role),pd.read_parquet(f_ker))
assert FSb==FSb2 and SY==SY2, 'dev/eval feature lists differ'
ROLE=[c for c in FSb if c.startswith('x_')]; KER=[c for c in FSb if c.startswith('k_')]; NEW=[c for c in FSb if c not in C.FEATURES and c not in ROLE and c not in KER]
VIEWS={'all':FSb,'all+sym':FSb+SY,'kern':KER+['ts'],'gplay':C.FEATURES+NEW,'role':ROLE+NEW}
CFGS={'lgb_base':('all',dict(C.PARAMS)),'lgb_sym':('all+sym',dict(C.PARAMS)),
 'lgb_deep':('all+sym',{**C.PARAMS,'num_leaves':63,'min_child_samples':20,'n_estimators':500,'learning_rate':.03}),
 'lgb_shal':('all+sym',{**C.PARAMS,'num_leaves':7,'n_estimators':900,'learning_rate':.02,'min_child_samples':60}),
 'lgb_goss':('all+sym',{**C.PARAMS,'boosting_type':'goss','num_leaves':31,'n_estimators':500}),
 'lgb_dart':('all+sym',{**C.PARAMS,'boosting_type':'dart','n_estimators':400,'learning_rate':.06,'drop_rate':.1}),
 'lgb_extra':('all+sym',{**C.PARAMS,'extra_trees':True,'num_leaves':31,'n_estimators':600}),
 'lgb_col3':('all+sym',{**C.PARAMS,'colsample_bytree':.3,'num_leaves':31,'n_estimators':600,'random_state':7}),
 'lgb_kern':('kern',dict(C.PARAMS)),'lgb_gplay':('gplay',dict(C.PARAMS)),'lgb_role':('role',dict(C.PARAMS)),
 'lgb_l2':('all+sym',{**C.PARAMS,'reg_lambda':200,'num_leaves':31,'n_estimators':600,'random_state':99}),
 'lgb_bag':('all+sym',{**C.PARAMS,'subsample':.6,'subsample_freq':1,'num_leaves':31,'n_estimators':600,'random_state':123}),
 'cat':('all+sym','catboost')}
g1=pd.read_parquet(f'{O}/r4/g1_evidence_rank.parquet')[['h','pair_id','evidence_rank','chron']].sort_values(['pair_id','evidence_rank'])
g1['run']=np.concatenate([runs(g.chron.values) for _,g in g1.groupby('pair_id',sort=False)]); g1['nruns']=g1.groupby('pair_id').run.transform('max')+1
s=s.merge(g1[['h','run','nruns']],on='h',how='left').sort_values(['slot','ts','h'],kind='stable').reset_index(drop=True); e=s[s.ev.astype(bool)]
FSfull=VIEWS['all+sym']
if fam=='coordinated_isolation': typB=(s.pa_at_trig<6).values
else:
    two=e[e.nruns==2]; clf=lgb.LGBMClassifier(n_estimators=200,learning_rate=.05,num_leaves=7,min_child_samples=10,colsample_bytree=.5,verbosity=-1,n_jobs=NJ).fit(two[FSfull].astype(float),(two.run==1).astype(int))
    typB=np.where(s.nruns.values==2,s.run.values==1,clf.predict_proba(s[FSfull].astype(float))[:,1]>0.5)
s['isA']=s.ev.astype(bool)&~typB; s['isB']=s.ev.astype(bool)&typB
nA=s.groupby('slot').isA.transform('sum'); nB=s.groupby('slot').isB.transform('sum'); nev=s.groupby('slot').ev.transform('sum')
lastA=s.slot.map(s[s.isA].groupby('slot').ts.max()); lastB=s.slot.map(s[s.isB].groupby('slot').ts.max())
incA=((nA<5)|(s.ts<=lastA)).values; incB=((nev<5)|((nB>0)&(s.ts<=lastB))).values
print(f'{fam}: dev rows {len(s)} A {int(s.isA.sum())} B {int(s.isB.sum())}; eval rows {len(ev)} pairs {ev.slot.nunique()}',flush=True)
lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9))); sg=lambda x: 1/(1+np.exp(-x))
def mk(p,seed):
    if p=='catboost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=600,depth=6,learning_rate=.05,l2_leaf_reg=6,verbose=0,thread_count=NJ,random_seed=seed)
    return lgb.LGBMClassifier(**{**p,'n_jobs':NJ,'random_state':seed})
LA_=[];LB_=[];used=[]
for cfg,(view,par) in CFGS.items():
    FS=[c for c in VIEWS[view] if c in s.columns]
    for seed in (260919,11,29):
        mA=mk(par,seed).fit(s.loc[incA,FS].astype(float),s.loc[incA,'isA'].astype(int)); LA_.append(lg(mA.predict_proba(ev[FS].astype(float))[:,1]))
        mB=mk(par,seed).fit(s.loc[incB,FS].astype(float),s.loc[incB,'isB'].astype(int)); LB_.append(lg(mB.predict_proba(ev[FS].astype(float))[:,1]))
    used.append(cfg); print(f'  {cfg} done ({len(used)}/{len(CFGS)})',flush=True)
GA=float(os.environ.get('GA',1.0)); GB=float(os.environ.get('GB',1.0))
pA=np.clip(GA*sg(np.mean(LA_,axis=0)),1e-9,1-1e-6); pB=np.clip(GB*sg(np.mean(LB_,axis=0)),1e-9,1-1e-6)
L,_,_=decode(ev,pA,pB); print(f'eval sum pA/pair median {pd.Series(pA).groupby(ev.slot.values).sum().median():.2f} pB {pd.Series(pB).groupby(ev.slot.values).sum().median():.2f}',flush=True)
cand=pd.read_parquet(f_cand); j=C.rank_candidates(ev,cand,L,weight=PAR if MODE=='rank' else 0.5)
if MODE=='stack':
    tab=pd.read_csv(os.environ['POKER_TABICL_EVAL']).rename(columns={'score':'tab'})
    j=j.merge(tab[['slot','hand_id','tab']],on=['slot','hand_id'],how='left',validate='one_to_one'); assert j.tab.notna().all()
    j['newscore']=lg(j.tab.values)+PAR*lg(j.q.values)
z=j.sort_values(['slot','newscore','ts','h'],ascending=[True,False,True,True],kind='stable').groupby('slot',sort=False).head(5).copy()
z['rank']=z.groupby('pair_id').cumcount()+1; pt=z.pivot(index='pair_id',columns='rank',values='hand_id'); pt.columns=[f'evidence_hand_{i}' for i in pt.columns]; out=pt.reset_index()
assert out.notna().all().all() and len(out)==cand.pair_id.nunique()
out.to_csv(out_csv,index=False); ev[['slot','h']].assign(pA=pA,pB=pB,L=L).to_parquet(out_csv.replace('.csv','_handprobs.parquet'))
old5=cand[cand.r<=5].groupby('pair_id').hand_id.apply(set); ov=[len(set(out.set_index('pair_id').loc[p_].values)&old5[p_]) for p_ in old5.index]
rec=dict(family=fam,mode=MODE,par=PAR,ga=GA,gb=GB,configs=used,models=len(LA_),features_all_sym=len(VIEWS['all+sym']),pairs=len(out),mean_overlap_with_R15_top5=float(np.mean(ov)),pairs_changed_vs_R15=int(np.sum(np.array(ov)<5)),sha256=hashlib.sha256(open(out_csv,'rb').read()).hexdigest())
json.dump(rec,open(out_csv.replace('.csv','.receipt.json'),'w'),indent=1); print(json.dumps(rec))
