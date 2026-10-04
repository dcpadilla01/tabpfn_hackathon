import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
tt = agent_api.train_targets()
base = agent_api.load_saved("e013_stock.parquet")
keys=["household_key","snapshot_day"]
df = tt.merge(base,on=keys,how="left")
cols0=[c for c in base.columns if c not in keys]
for c in cols0:
    if df[c].dtype=='object' or str(df[c].dtype)=='category':
        df[c]=df[c].astype('object').astype('category').cat.codes.astype(float).replace(-1,np.nan)
print("x_exp4w describe:", df.x_exp4w.describe().round(2).to_dict())
print("recency describe:", df.recency.describe().round(1).to_dict())

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

m_ref=ridge_eval(df,cols0,"ref e013")

def hinges(frame, col, edges, prefix):
    x = frame[col].fillna(0).clip(lower=0).values
    out = {}
    for e in edges:
        out[f"{prefix}h{int(e)}"] = np.maximum(0.0, x - e)
    return pd.DataFrame(out, index=frame.index)

# hinges on key level features (edges chosen from spend distribution)
edges_lvl = [10, 25, 50, 100, 200, 400, 800]
H = pd.concat([
    hinges(df,'x_exp4w',edges_lvl,'hx_'),
    hinges(df,'spend_84',edges_lvl,'hs84_'),
    hinges(df,'x_spend_p1',edges_lvl,'hp1_'),
    hinges(df,'x_spend_p2',edges_lvl,'hp2_'),
    hinges(df,'recency',[7,14,28,56,112],'hr_'),
], axis=1)
df2 = pd.concat([df, H], axis=1)
ridge_eval(df2, cols0+list(H.columns), "e013+hinges5", verbose=True)
# hinges only on x_exp4w
H1 = hinges(df,'x_exp4w',edges_lvl,'hx_')
df3 = pd.concat([df,H1],axis=1)
ridge_eval(df3, cols0+list(H1.columns), "e013+hinge_exp")
# fewer hinges
H2 = pd.concat([hinges(df,'x_exp4w',[25,100,300],'hx_'), hinges(df,'recency',[14,28],'hr_')],axis=1)
df4 = pd.concat([df,H2],axis=1)
ridge_eval(df4, cols0+list(H2.columns), "e013+hinges_few")
