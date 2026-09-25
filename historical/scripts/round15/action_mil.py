"""Raw decision -> policy pretraining -> action encoder -> hand ranking.
No network, no Kaggle submission. Supports pack, cv, fit and predict commands.
Raw GB10 runs are required: tests on constructed records are not score evidence.
"""
from __future__ import annotations
import argparse, json, importlib, sys, math, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


def logbb(v, bb): return math.log1p(max(float(v), 0.) / bb)


def encode_record(r, A, B, step, total_steps):
    """q0 context excludes current action/amount and all non-actor private cards."""
    i=int(r['i']); bb=float(r['bb']); st=int(r['st']);
    if bb<=0 or st not in range(4):raise ValueError('Invalid betting state')
    geom=np.asarray(r['geom'],float);alive=np.asarray(r['alive'],bool);can=np.asarray(r['can'],bool)
    role=0 if i==A else 1 if i==B else 2
    partner=B if i==A else A if i==B else None
    toward=partner is not None and int(r['la'])==partner and float(r['tc'])>0
    other=(B if i==A else A) if partner is not None else A
    # Own information: made hand, draws from board prefix, pot, price, stack,
    # public live sets, actual prior raises, legal opportunity proxy.
    context=np.r_[geom[i], np.eye(4)[st],
       logbb(r['pot'],bb),logbb(r['tc'],bb),logbb(r['stack'],bb),
       float(r['tc'])/max(float(r['pot'])+float(r['tc']),1e-8),
       alive.mean(),can.mean(),float(r['raise_proxy']),float(r.get('raises',0))/10,
       logbb(np.asarray(r['total'])[i],bb),
       ((i-int(r.get('btn',0)))%6)/5 if 'btn' in r else 0.,float('btn' in r)]
    yy=int(r['y'])
    if yy not in range(4):raise ValueError('Invalid action class')
    hu=np.asarray(r.get('hu',np.full((6,6),-1.)),float)
    q=float(hu[i,partner]) if partner is not None else -1.
    known=float(0<=q<=1)
    pg=geom[partner] if partner is not None else np.zeros(10)
    pair_geom=np.r_[geom[A],geom[B]]
    faced=np.asarray(r.get('faced',np.zeros(6)),int)
    same=sum((int(faced[o])&((1<<A)|(1<<B)))==((1<<A)|(1<<B)) for o in range(6) if o not in (A,B))
    # Current action and retrospective partner cards are allowed for evidence;
    # they never enter the q0 action-prediction input.
    observed=np.r_[context,np.eye(4)[yy],np.eye(3)[role],
       logbb(r['amt'],bb),float(r['amt'])/max(bb,float(r['pot'])),
       float(toward),float(alive[A]),float(alive[B]),float(can[A]),float(can[B]),
       known,q if known else 0.,pg,pair_geom,same/4,
       step/max(total_steps-1,1)]
    # Position is within a betting sequence, not across hands / file order.
    return np.asarray(context,np.float32),np.asarray(observed,np.float32),yy


def pack_records(records_path, meta_path, out):
    d=pd.read_csv(meta_path).reset_index(drop=True)
    if d.empty:raise ValueError('Empty candidate metadata')
    if d.duplicated(['slot','hand_id']).any():raise ValueError('Duplicate keys')
    records={}
    with open(records_path) as f:
        for line in f:
            z=json.loads(line);key=(str(z['slot']),str(z['hand_id']))
            if key in records:raise ValueError('Duplicate record key')
            records[key]=z
    encoded=[]; maxlen=0
    for row in d.itertuples():
        z=records[(str(row.slot),str(row.hand_id))];rr=z['records'];A,B=map(int,z['seats'])
        if not rr:raise ValueError('Empty action sequence')
        views=[]
        for a,b in [(A,B),(B,A)]:
            views.append([encode_record(r,a,b,k,len(rr)) for k,r in enumerate(rr)])
        encoded.append(views);maxlen=max(maxlen,len(rr))
    C=len(encoded[0][0][0][0]);D=len(encoded[0][0][0][1]);N=len(d)
    ctx=np.zeros((N,2,maxlen,C),np.float32);x=np.zeros((N,2,maxlen,D),np.float32)
    mask=np.zeros((N,2,maxlen),bool);y=np.full((N,2,maxlen),-1,np.int64)
    for n,views in enumerate(encoded):
        for v,rows in enumerate(views):
            for k,(c,xx,yy) in enumerate(rows):ctx[n,v,k]=c;x[n,v,k]=xx;y[n,v,k]=yy;mask[n,v,k]=True
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'actions.npz',context=ctx,x=x,mask=mask,action=y)
    d.to_csv(out/'meta.csv',index=False)
    receipt={'rows':N,'pairs':int(d.slot.nunique()),'max_actions':maxlen,'context_dim':C,'event_dim':D,
       'source_records_sha256':hashlib.sha256(Path(records_path).read_bytes()).hexdigest(),
       'raw_training_executed':False,'q0_current_action_excluded':True,
       'no_pair_or_hand_identifiers_in_model':True,'orientation':'average predictions'}
    (out/'pack.json').write_text(json.dumps(receipt,indent=2));return receipt


class ActionMIL(nn.Module):
    def __init__(self,c,d,width=64,sequence=False):
        super().__init__();self.width=width;self.sequence=sequence
        self.own=nn.Sequential(nn.Linear(c,width),nn.GELU(),nn.Linear(width,width),nn.GELU())
        self.policy=nn.Linear(width,4)
        self.event=nn.Sequential(nn.Linear(d+width+4,width),nn.GELU(),nn.LayerNorm(width),
             nn.Linear(width,width),nn.GELU())
        self.gru=nn.GRU(width,width,batch_first=True) if sequence else None
        self.av=nn.Linear(width,width);self.au=nn.Linear(width,width);self.aw=nn.Linear(width,1)
        self.head=nn.Sequential(nn.Linear(width*2+1,width),nn.GELU(),nn.Linear(width,1))
    def policy_logits(self,c):return self.policy(self.own(c))
    def forward(self,c,x,mask):
        # The two endpoint orderings must both be passed by the caller.
        n,v,t,_=x.shape;c=c.reshape(n*v,t,-1);x=x.reshape(n*v,t,-1);mask=mask.reshape(n*v,t)
        if not mask.any(1).all():raise ValueError('Empty bag')
        h0=self.own(c);prob=torch.softmax(self.policy(h0),-1)
        h=self.event(torch.cat([x,h0,prob],-1))
        if self.gru is not None:h,_=self.gru(h)
        att=self.aw(torch.tanh(self.av(h))*torch.sigmoid(self.au(h))).squeeze(-1)
        att=att.masked_fill(~mask,-torch.inf).softmax(-1)
        mean=(att[...,None]*h).sum(1)
        maximum=h.masked_fill(~mask[...,None],-torch.inf).max(1).values
        # Counts are gameplay opportunity counts; no cross-hand chronological index.
        count=mask.sum(1).float().log1p()[:,None]
        score=self.head(torch.cat([mean,maximum,count],-1)).reshape(n,v).mean(1)
        return score


def pair_loss(scores, labels, groups):
    losses=[]
    for g in torch.unique(groups):
        ix=groups==g;s=scores[ix];y=labels[ix];p=s[y==1];u=s[y==0]
        if len(p) and len(u):
            # Annotation retrieval target: y=0 is "not in reference", not normal play.
            losses.append(F.softplus(u[:,None]-p[None,:]).mean()+.1*F.binary_cross_entropy_with_logits(s,y.float()))
    if not losses:raise ValueError('No contrastive evidence group')
    return torch.stack(losses).mean()


def normalize_fit(values,mask,train):
    valid=values[train][mask[train]]
    mu=valid.mean(0);sd=np.maximum(valid.std(0),1e-3)
    return mu.astype(np.float32),sd.astype(np.float32)


def apply_norm(x,mask,mu,sd):
    z=np.clip((x-mu)/sd,-10,10);z[~mask]=0
    if not np.isfinite(z).all():raise ValueError('Invalid input')
    return z.astype(np.float32)


def train_one(d,raw,train,device,seed,epochs,policy_epochs,batch_pairs,sequence,background=None):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    c,x,mask,actions=[raw[k] for k in ['context','x','mask','action']]
    cm,cs=normalize_fit(c,mask,train);xm,xs=normalize_fit(x,mask,train)
    cc=apply_norm(c,mask,cm,cs);xx=apply_norm(x,mask,xm,xs)
    model=ActionMIL(c.shape[-1],x.shape[-1],sequence=sequence).to(device)
    # Deduplicate same hand/action across candidate pair views. Actor context
    # has no pair-dependent fields and is identical in the two orientations.
    seen=set();policy_c=[];policy_y=[]
    for i in np.flatnonzero(train):
        for t in np.flatnonzero(mask[i,0]):
            key=(str(d.hand_id.iloc[i]),int(t))
            if key not in seen:seen.add(key);policy_c.append(cc[i,0,t]);policy_y.append(actions[i,0,t])
    if background is not None:
        # The caller must pass a background already excluding validation pools.
        bc,by=background;policy_c.extend(np.clip((bc-cm)/cs,-10,10));policy_y.extend(by)
    pc=torch.tensor(np.asarray(policy_c),device=device);py=torch.tensor(np.asarray(policy_y),device=device)
    opt=torch.optim.AdamW(list(model.own.parameters())+list(model.policy.parameters()),lr=1e-3,weight_decay=1e-3)
    for ep in range(policy_epochs):
        order=rng.permutation(len(pc))
        for start in range(0,len(pc),512):
            ix=torch.as_tensor(order[start:start+512],device=device)
            loss=F.cross_entropy(model.policy_logits(pc[ix]),py[ix]);opt.zero_grad();loss.backward();opt.step()
    # Freeze the causal policy backbone so retrieval cannot make its output
    # resemble evidence labels. Event encoder still learns rich observed actions.
    for mod in [model.own,model.policy]:
        for par in mod.parameters():par.requires_grad=False
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3,weight_decay=.01)
    slots=d.slot.to_numpy();unique=np.unique(slots[train]);yy=d.ev.to_numpy(int)
    tensors=[torch.tensor(v,device=device) for v in [cc,xx,mask]]
    history=[]
    for ep in range(epochs):
        model.train();shuffled=rng.permutation(unique);total=0.;steps=0
        for start in range(0,len(shuffled),batch_pairs):
            chosen=shuffled[start:start+batch_pairs]
            ix=np.flatnonzero(train & np.isin(slots,chosen))
            group=np.unique(slots[ix],return_inverse=True)[1]
            ii=torch.tensor(ix,device=device)
            pred=model(*[v[ii] for v in tensors]);loss=pair_loss(pred,torch.tensor(yy[ix],device=device),torch.tensor(group,device=device))
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
            total+=float(loss.detach());steps+=1
        history.append(total/max(steps,1))
        if (ep+1)%10==0:print(f'epoch {ep+1} retrieval_loss={history[-1]:.6f}',flush=True)
    return model,{'cm':cm,'cs':cs,'xm':xm,'xs':xs},history


def predict(model,raw,norm,device):
    model.eval();mask=raw['mask'];cc=apply_norm(raw['context'],mask,norm['cm'],norm['cs']);xx=apply_norm(raw['x'],mask,norm['xm'],norm['xs'])
    out=[]
    with torch.no_grad():
        for start in range(0,len(xx),128):
            tensors=[torch.tensor(v[start:start+128],device=device) for v in [cc,xx,mask]]
            out.extend(model(*tensors).cpu().numpy())
    return np.asarray(out)


def metric(d,s):
    vals=[]
    for slot,ix in d.groupby('slot',sort=False).indices.items():
        g=d.iloc[ix];m=int(g.m_p.iloc[0]);y=g.ev.to_numpy(int)[np.argsort(-s[ix],kind='stable')][:5]
        vals.append({'slot':slot,'pool':g.pool.iloc[0],'fold':g.fold.iloc[0],
          'E':float(np.sum(y*np.cumsum(y)/np.arange(1,len(y)+1))/min(m,5))})
    return pd.DataFrame(vals)


def main_run(a):
    root=Path(a.pack);d=pd.read_csv(root/'meta.csv');raw=dict(np.load(root/'actions.npz',allow_pickle=False))
    if not {'slot','hand_id'}.issubset(d) or d.duplicated(['slot','hand_id']).any():raise ValueError('Candidate key error')
    if a.command!='predict' and (not {'pool','fold'}.issubset(d) or d.groupby('pool').fold.nunique().max()!=1):raise ValueError('Training pool/fold error')
    if len(d)!=len(raw['x']):raise ValueError('Length error')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);device=torch.device(a.device)
    torch.set_num_threads(a.threads)
    if a.command=='predict':
        if not a.checkpoint:raise ValueError('predict requires the self-produced --checkpoint')
        ck=torch.load(a.checkpoint,map_location=device,weights_only=False)
        model=ActionMIL(raw['context'].shape[-1],raw['x'].shape[-1],sequence=ck['sequence']).to(device)
        model.load_state_dict(ck['model']);s=predict(model,raw,ck['norm'],device)
    else:
        if not {'ev','m_p','fold','pool'}.issubset(d):raise ValueError('Training labels/metadata missing')
        if not d.ev.isin([0,1]).all() or d.m_p.le(0).any():raise ValueError('cv/fit expects reference-labelled positive pairs')
        fs=sorted(d.fold.unique()) if a.command=='cv' else [None];s=np.zeros(len(d));hist={}
        for f in fs:
            tr=np.ones(len(d),bool) if f is None else d.fold.ne(f).to_numpy()
            bg=None
            if a.background:
                b=dict(np.load(a.background,allow_pickle=False));valid=np.ones(len(b['context']),bool) if f is None else b['fold']!=f
                bg=(b['context'][valid],b['action'][valid])
            model,norm,history=train_one(d,raw,tr,device,a.seed,a.epochs,a.policy_epochs,a.batch_pairs,a.sequence,bg)
            p=predict(model,raw,norm,device);s[:]=p if f is None else s;s[~tr]=p[~tr]
            if f is None:s=p
            hist[str(f)]=history
            torch.save({'model':model.state_dict(),'norm':norm,'sequence':bool(model.sequence),'fold':f,'seed':a.seed},out/f'model_{f}.pt')
            print('completed fold',f,flush=True)
        (out/'training_history.json').write_text(json.dumps(hist))
    keep=[c for c in ['slot','hand_id','pool','fold','ev','m_p'] if c in d]
    d[keep].assign(score=s).to_csv(out/'predictions.csv.gz',index=False)
    rep={'state':'EXECUTED_ON_INPUT_PACK','command':a.command,'seed':a.seed,
        'baseline_upstream_not_used_in_model':True,'background':a.background,
        'no_submission':True,'input_pack':str(root),'sequence':bool(model.sequence),
        'validation_scope':'pool-separated raw learner; candidate-selection/upstream provenance still requires confirmation'}
    if a.command=='cv':
        pm=metric(d,s);pm.to_csv(out/'pair_metrics.csv',index=False);rep['E']=float(pm.E.mean())
    (out/'results.json').write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('pack');b.add_argument('--records',required=True);b.add_argument('--meta',required=True);b.add_argument('--out',required=True)
    for command in ['cv','fit','predict']:
        s=sub.add_parser(command)
        for k in ['pack','out']:s.add_argument('--'+k,required=True)
        s.add_argument('--device',default='cuda');s.add_argument('--seed',type=int,default=20260918)
        s.add_argument('--epochs',type=int,default=40);s.add_argument('--policy-epochs',type=int,default=8)
        s.add_argument('--batch-pairs',type=int,default=8);s.add_argument('--threads',type=int,default=8)
        s.add_argument('--sequence',action='store_true');s.add_argument('--background');s.add_argument('--checkpoint')
    a=p.parse_args()
    if a.command=='pack':print(json.dumps(pack_records(a.records,a.meta,a.out),indent=2))
    else:main_run(a)
