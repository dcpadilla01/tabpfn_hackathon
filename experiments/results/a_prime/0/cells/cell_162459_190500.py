import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
tt = agent_api.train_targets()
base = agent_api.load_saved("e013_stock.parquet")
cal  = agent_api.load_saved("cal_v1_offline.parquet")
keys=["household_key","snapshot_day"]
df = tt.merge(base,on=keys,how="left").merge(cal,on=keys,how="left")
cols0=[c for c in base.columns if c not in keys]
for c in cols0:
    if df[c].dtype=='object' or str(df[c].dtype)=='category':
        df[c]=df[c].astype('object').astype('category').cat.codes.astype(float).replace(-1,np.nan)

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
        print("   zeros meanpred=%.1f MAE=%.1f | actives MAE=%.1f"%(ph[yh0].mean(),np.abs(r[yh0]).mean(),np.abs(r[~yh0]).mean()))
    print(f"[{tag}] alpha={a} inner={best[1]:.2f} hold431={mae:.3f}")
    return mae

m_ref=ridge_eval(df,cols0,"ref e013")
for add in [["cal1"],["cal1","cal2"],["cal_bl"],["cal1","cal2","cal_bl"]]:
    ridge_eval(df,cols0+add,"e013+"+",".join(add))
# calibration features ONLY combined with a few core levels (small model)
core=[c for c in cols0 if c in ['x_exp4w','spend_28','spend_84','spend_364','baskets_28','recency','x_spend_p1','x_spend_p2','active_28','x_bv_4']]
ridge_eval(df,core+["cal1","cal2","cal_bl"],"core+cal")
ridge_eval(df,["cal1","cal2","cal_bl"],"cal only",verbose=True)
ridge_eval(df,core,"core only")
# ratio to x_exp4w
df['cal_ratio']=df['cal_bl']/df['x_exp4w'].replace(0,np.nan)
df['cal_div']=df['cal_bl']-df['x_exp4w'].fillna(0)
ridge_eval(df,cols0+["cal1","cal2","cal_bl","cal_ratio","cal_div"],"e013+cal+ratios",verbose=True)
