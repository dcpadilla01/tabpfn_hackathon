import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
tt = agent_api.train_targets()
base = agent_api.load_saved("e013_stock.parquet")
dept = agent_api.load_saved("dept_v1_offline.parquet")
keys=["household_key","snapshot_day"]

def enc(frame, cols):
    for c in cols:
        if frame[c].dtype=='object' or str(frame[c].dtype)=='category':
            frame[c]=frame[c].astype('object').astype('category').cat.codes.astype(float).replace(-1,np.nan)
    return frame

def ridge_eval(frame, feat_cols, tag, alphas=(32,64,128,256,512), verbose=False):
    fit=frame[frame.snapshot_day<=403]; hold=frame[frame.snapshot_day==431]
    yf=fit.future_spend_4w.values; yh=hold.future_spend_4w.values
    Xf=fit[feat_cols].astype(float).values; Xh=hold[feat_cols].astype(float).values
    med=np.nanmedian(Xf,axis=0); Xf=np.where(np.isnan(Xf),med,Xf); Xh=np.where(np.isnan(Xh),med,Xh)
    mu=Xf.mean(0); sd=Xf.std(0)+1e-9; Xf=(Xf-mu)/sd; Xh=(Xh-mu)/sd
    Xf=np.hstack([Xf,np.ones((len(Xf),1))]); Xh=np.hstack([Xh,np.ones((len(Xh),1))])
    itr=(fit.snapshot_day<=375).values; iva=(fit.snapshot_day==403).values
    best=None
    for a in alphas:
        A=Xf[itr]; w=np.linalg.solve(A.T@A+a*np.eye(A.shape[1]),A.T@yf[itr])
        mae=np.abs(np.clip(Xf[iva]@w,0,None)-yf[iva]).mean()
        if best is None or mae<best[1]: best=(a,mae)
    a=best[0]; G=Xf.T@Xf+a*np.eye(Xf.shape[1]); w=np.linalg.solve(G,Xf.T@yf)
    ph=np.clip(Xh@w,0,None); mae=np.abs(ph-yh).mean()
    if verbose:
        r=ph-yh; yh0=yh==0
        print("   zeros meanpred=%.1f | actives MAE=%.1f"%(ph[yh0].mean(),np.abs(r[~yh0]).mean()))
    print(f"[{tag}] alpha={a} inner={best[1]:.2f} hold431={mae:.3f}")
    return mae

cols0=[c for c in base.columns if c not in keys]
dfD = enc(tt.merge(base,on=keys,how="left").merge(dept,on=keys,how="left"), cols0+[c for c in dept.columns if c not in keys+['index']])
dcols=[c for c in dept.columns if c not in keys+['index']]
ridge_eval(dfD,cols0,"ref")
ridge_eval(dfD,cols0+dcols,"e013+dept",verbose=True)

# ---------------- two-part model from lagged pairs ----------------
v=agent_api.snapshot()
t=v.transactions[['household_key','day','sales_value']]
first=t.groupby('household_key').day.min().rename('first_day')
def spend_win(lo,hi,pool=None):
    s=t[(t.day>=lo)&(t.day<=hi)].groupby('household_key').sales_value.sum()
    if pool is not None: s=s.reindex(pool)
    return s.fillna(0.0)
def hist_feats(a, pool):
    tp=t[t.day<=a]
    last=tp.groupby('household_key').day.max()
    rec=(a-last).reindex(pool).astype(float)
    tw=tp[(tp.day>=a-83)].copy()
    tw['w']=np.exp(-(a-tw.day)/40.0)
    xexp=tw.assign(sv=tw.sales_value*tw.w).groupby('household_key').sv.sum().reindex(pool).fillna(0.0)
    pos=np.zeros(len(pool)); zs=np.zeros(len(pool))
    bl=[]
    for k in range(1,13):
        bl.append(spend_win(a-28*k+1,a-28*(k-1),pool).values)
    B=np.array(bl).T
    pos=(B>0).mean(axis=1)
    zs=np.zeros(len(pool))
    for i in range(len(pool)):
        c=0
        for k in range(12):
            if B[i,k]==0: c+=1
            else: break
        zs[i]=c
    return pd.DataFrame({'rec':rec.values,'xexp':xexp.values,'pos':pos,'zst':zs},index=pool)

EDGES=[0,10,25,50,75,100,150,200,300,450,600]
def fit_logistic(X,y,lam=1.0,iters=25):
    mu=X.mean(0); sd=X.std(0)+1e-9; Z=(X-mu)/sd; Z=np.hstack([Z,np.ones((len(Z),1))])
    w=np.zeros(Z.shape[1])
    for _ in range(iters):
        p=1/(1+np.exp(-Z@w))
        g=Z.T@(p-y)-lam*w; H=Z.T@(Z*(p*(1-p))[:,None])+lam*np.eye(Z.shape[1])
        step=np.linalg.solve(H,g); w=w-step
        if np.abs(step).max()<1e-6: break
    return w,mu,sd
rows=[]
days=[95,123,151,179,207,235,263,291,319,347,375,403,431,459]
for d in days:
    elig=first[first<=d-84].index
    parts=[]
    for lag,ylo,yhi in [(28,d-27,d),(56,d-55,d-28)]:
        a=d-lag
        pool=first[first<=a-84].index
        if len(pool)<200:
            pool=first[first<=a-28].index
        if len(pool)<200: parts.append(None); continue
        Hf=hist_feats(a,pool)
        yact=(spend_win(ylo,yhi,pool).values>0).astype(float)
        yspend=spend_win(ylo,yhi,pool).values
        w,mu,sd=fit_logistic(Hf[['rec','xexp','pos']].values,yact)
        Hc=hist_feats(d,elig)
        Z=(Hc[['rec','xexp','pos']].values-mu)/sd; Z=np.hstack([Z,np.ones((len(Z),1))])
        p=np.clip(1/(1+np.exp(-Z@w)),0.02,0.98)
        act=yspend>0
        b=np.digitize(Hc['xexp'].values,EDGES)
        cal=np.full(len(elig),np.nan)
        for bi in np.unique(b):
            m=b==bi
            sel=(np.digitize(Hf['xexp'].values,EDGES)==bi)&act
            if sel.sum()>=30: cal[m]=yspend[sel].mean()
        parts.append(pd.DataFrame({'p_act':p,'cal':cal},index=elig))
    f=pd.DataFrame(index=elig)
    p1=parts[0]['p_act']; c1=parts[0]['cal']
    p2=parts[1]['p_act'] if parts[1] is not None else p1
    c2=parts[1]['cal'] if parts[1] is not None else c1
    f['tp_p']=((p1+p2)/2).values
    calbl=pd.concat([c1,c2],axis=1).mean(axis=1)
    f['tp_f']= (f['tp_p']*calbl.fillna(0)).values
    f['tp_px']= (f['tp_p']*base.set_index(keys).reindex(pd.MultiIndex.from_arrays([elig,[d]*len(elig)]))['x_exp4w'].fillna(0)).values if False else (f['tp_p'].values*0)
    rows.append(f.reset_index().rename(columns={'index':'household_key'}).assign(snapshot_day=d))
tp=pd.concat(rows,ignore_index=True)
print("tp",tp.shape, tp.tp_p.describe().round(3).to_dict())
agent_api.save_table(tp,"tp_v1_offline.parquet")
dfT=enc(tt.merge(base,on=keys,how="left").merge(tp,on=keys,how="left"),cols0)
tcols=[c for c in tp.columns if c not in keys+['index']]
ridge_eval(dfT,cols0+tcols,"e013+twopart",verbose=True)
ridge_eval(dfT,cols0+["tp_f"],"e013+tp_f")
ridge_eval(dfT,cols0+["tp_p"],"e013+tp_p")
