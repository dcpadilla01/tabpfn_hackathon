import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
tt = agent_api.train_targets()
base = agent_api.load_saved("e013_stock.parquet")
shape = agent_api.load_saved("shape_v1_offline.parquet")
keys=["household_key","snapshot_day"]
df2 = tt.merge(base,on=keys,how="left").merge(shape,on=keys,how="left")
cols0=[c for c in base.columns if c not in keys]
for c in cols0:
    if df2[c].dtype=='object' or str(df2[c].dtype)=='category':
        df2[c]=df2[c].astype('object').astype('category').cat.codes.astype(float).replace(-1,np.nan)
scols=[c for c in shape.columns if c not in keys+['index']]

def ridge_eval(frame, feat_cols, tag, alphas=(8,16,32,64,128), verbose=False):
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

m_ref=ridge_eval(df2,cols0,"ref")
ridge_eval(df2,cols0+scols,"e013+shape",verbose=True)
ridge_eval(df2,scols,"shape only")
core=[c for c in cols0 if c in ['x_exp4w','spend_28','spend_84','spend_364','baskets_28','recency','x_spend_p1','x_spend_p2','active_28','x_bv_4','x_r_4_8']]
ridge_eval(df2,core+scols,"core+shape")
ridge_eval(df2,core,"core")

# ---- GBM ceiling check: tiny histogram GBM on e013 features ----
def gbm_eval(frame, feat_cols, tag, n_rounds=400, lr=0.05, depth=6, seed=0):
    fit=frame[frame.snapshot_day<=403]; hold=frame[frame.snapshot_day==431]
    yf=fit.future_spend_4w.values; yh=hold.future_spend_4w.values
    def prep(X):
        X=X.astype(float).copy()
        med=np.nanmedian(X,axis=0)
        return np.where(np.isnan(X),med,X)
    Xf=prep(fit[feat_cols].values); Xh=prep(hold[feat_cols].values)
    # log-target GBM
    lf=np.log1p(yf)
    pred_f=np.zeros(len(fit)); pred_h=np.zeros(len(hold))
    feat_idx=np.arange(Xf.shape[1])
    rng=np.random.RandomState(seed)
    nfeat=min(40,len(feat_idx))
    for it in range(n_rounds):
        r=lf-pred_f
        feats=rng.choice(feat_idx,nfeat,replace=False)
        best=None
        for j in feats:
            order=np.argsort(Xf[:,j])
            xs=Xf[order,j]; rs=r[order]
            # candidate thresholds at quantiles
            qs=np.unique(np.quantile(xs,[0.1,0.25,0.5,0.75,0.9]))
            for q in qs:
                m=xs<=q
                if m.sum()<50 or (~m).sum()<50: continue
                gl=rs[m].sum(); gr=rs[~m].sum()
                gain=gl*gl/(m.sum()+1e-9)+gr*gr/((~m).sum()+1e-9)
                if best is None or gain>best[0]: best=(gain,j,q,m)
        if best is None: break
        _,j,q,m=best
        lv=r[m].mean(); rv=r[~m].mean()
        pred_f+=lr*np.where(m,lv,rv)
        pred_h+=lr*np.where(Xh[:,j]<=q,lv,rv)
    ph=np.clip(np.expm1(pred_h),0,None)
    mae=np.abs(ph-yh).mean()
    print(f"[GBM {tag}] hold431={mae:.3f}  zeros meanpred={ph[yh==0].mean():.1f}")
    return mae
gbm_eval(df2,cols0,"e013 log-target")
gbm_eval(df2,cols0+scols,"e013+shape log-target")
gbm_eval(df2,core+scols,"core+shape log-target")
