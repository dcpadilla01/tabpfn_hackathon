import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings('ignore')
d=agent_api.load_saved('e011_cand.parquet'); t=agent_api.train_targets()
d=d.merge(t,on=['household_key','snapshot_day'],how='inner')
print('cand cols sample:', [c for c in d.columns if 'index' in c])
E6=[c for c in agent_api.load_saved('e006_dynamics.parquet').columns if c not in ('household_key','snapshot_day')]
E6=[c+'__x' if c+'__x' in d.columns else c for c in E6]
E6=[c for c in E6 if c in d.columns and c!='index']
NEW=['sp28_lo','sp28_hi','ew28_lo','ew28_hi','ew28_lo25','ew28_hi25','lapsed','exp_spend','exp_spend2','rec_lo','rec_hi','act_ewma28','act_ewma56','act_share_13w','act_share_26w','act_share_52w']
NEW=[c for c in NEW if c in d.columns]
def to_mats(d_, cols):
    mats=[]
    for c in cols:
        s=d_[c]
        if s.dtype.kind not in 'iufcb': s=pd.Series(pd.factorize(s)[0],index=s.index)
        mats.append(pd.to_numeric(s,errors='coerce').to_numpy(dtype=float))
    return np.column_stack(mats)
y=d.future_spend_4w.to_numpy(float); day=d.snapshot_day.to_numpy(); days=sorted(set(day.tolist()))
def fit_pred(Xtr,ytr,Xte,lam):
    mu=np.nanmean(Xtr,axis=0); sd=np.nanstd(Xtr,axis=0); sd[sd==0]=1.0
    Ztr=np.nan_to_num((Xtr-mu)/sd); Zte=np.nan_to_num((Xte-mu)/sd)
    Ztr=np.hstack([Ztr,np.ones((len(Ztr),1))]); Zte=np.hstack([Zte,np.ones((len(Zte),1))])
    P=np.eye(Ztr.shape[1])*lam; P[-1,-1]=0.0
    return Zte@np.linalg.solve(Ztr.T@Ztr+P, Ztr.T@ytr)
def loso(cols_,lam=10):
    X=to_mats(d,cols_); errs=[]
    for k in days:
        tr=day!=k; te=day==k
        errs.append(np.abs(fit_pred(X[tr],y[tr],X[te],lam)-y[te]))
    return np.concatenate(errs).mean()
rng=np.random.default_rng(2); X=to_mats(d,E6)
imp={}
for j,c in enumerate(E6):
    dl=[]
    for k in days:
        tr=day!=k; te=day==k
        p0=fit_pred(X[tr],y[tr],X[te],10)
        Xp=X[te].copy(); Xp[:,j]=Xp[rng.permutation(len(Xp)),j]
        p1=fit_pred(X[tr],y[tr],Xp,10)
        dl.append(np.abs(p1-y[te]).mean()-np.abs(p0-y[te]).mean())
    imp[c]=float(np.mean(dl))
rank=sorted(E6,key=lambda c:imp[c],reverse=True)
print('ranks 25-50:',[(c,round(imp[c],2)) for c in rank[25:50]])
V={'top30':rank[:30],'top45':rank[:45],'top60':rank[:60]}
for name,base in V.items():
    for lam in [3,10,30,100]:
        print(name,'lam',lam,round(loso(base,lam),3))
print()
H1=['sp28_lo','sp28_hi','ew28_lo','ew28_hi','rec_lo','rec_hi']
H2=H1+['lapsed','act_ewma28','act_share_26w']
H3=H1+['lapsed','act_ewma28','act_share_26w','exp_spend','ew28_lo25','ew28_hi25']
for hn,H in [('H1',H1),('H2',H2),('H3',H3)]:
    for lam in [3,10,30,100]:
        print('top45+',hn,'lam',lam,round(loso(rank[:45]+H,lam),3))
