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

def ridge_eval(frame, feat_cols, tag, alphas=(8,16,32,64,128,256), verbose=False):
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

def hinges(frame, col, edges, prefix, fill=0.0):
    x = frame[col].fillna(fill).clip(lower=0).values.astype(float)
    return pd.DataFrame({f"{prefix}h{int(e)}": np.maximum(0.0, x-e) for e in edges}, index=frame.index)

H = pd.concat([
    hinges(df,'x_exp4w',[10,25,50,100,200,400,800],'hx_'),
    hinges(df,'spend_84',[25,100,300],'hs84_'),
    hinges(df,'x_spend_p1',[10,25,50,100,200,400,800],'hp1_'),
    hinges(df,'x_spend_p2',[10,25,50,100,200,400,800],'hp2_'),
    hinges(df,'recency',[7,14,28,56,112],'hr_'),
    hinges(df,'x_spend_12',[25,100,300],'hs12_'),
    hinges(df,'baskets_28',[1,3,6,10],'hb28_'),
    hinges(df,'x_bv_4',[10,25,50,100],'hbv_'),
    hinges(df,'days_28',[1,4,8,14],'hd28_'),
],axis=1)
dfH = pd.concat([df,H],axis=1)
ridge_eval(dfH,cols0,"ref")
ridge_eval(dfH,cols0+list(H.columns),"e013+hinges_ext",verbose=True)
# pruning: drop weak blocks
dropmix=[c for c in cols0 if c.startswith('p_') or c.startswith('coh_') or c.startswith('stk_') or c in ('dow_entropy','modal_dow','zero_w12','wk_cv','unit_price','n_prod84','p_other')]
keep=[c for c in cols0 if c not in dropmix]
ridge_eval(dfH,keep+list(H.columns),"pruned+hinges_ext",verbose=True)
ridge_eval(dfH,keep,"pruned")
