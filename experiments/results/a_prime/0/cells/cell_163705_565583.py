import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
tt = agent_api.train_targets()
base = agent_api.load_saved("e013_stock.parquet")
dept = agent_api.load_saved("dept_v1_offline.parquet")
struct = agent_api.load_saved("structure_v1.parquet")
keys=["household_key","snapshot_day"]

def enc(frame, cols):
    for c in cols:
        if c in frame.columns and (frame[c].dtype=='object' or str(frame[c].dtype)=='category'):
            frame[c]=frame[c].astype('object').astype('category').cat.codes.astype(float).replace(-1,np.nan)
    return frame

def ridge_eval(frame, feat_cols, tag, alphas=(64,128,256,512), verbose=False):
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
dfm = enc(tt.merge(base,on=keys,how="left").merge(dept,on=keys,how="left").merge(struct,on=keys,how="left"),
          cols0+[c for c in dept.columns if c not in keys+['index']]+[c for c in struct.columns if c not in keys+['index']])
dcols=[c for c in dept.columns if c not in keys+['index']]
scols=[c for c in struct.columns if c not in keys+['index']]

def hinges(frame, col, edges, prefix, fill=0.0):
    x = frame[col].fillna(fill).clip(lower=0).values.astype(float)
    return pd.DataFrame({f"{prefix}h{int(e)}": np.maximum(0.0, x-e) for e in edges}, index=frame.index)
H = pd.concat([
    hinges(dfm,'x_exp4w',[10,25,50,100,200,400,800],'hx_'),
    hinges(dfm,'spend_84',[25,100,300],'hs84_'),
    hinges(dfm,'x_spend_p1',[10,25,50,100,200,400,800],'hp1_'),
    hinges(dfm,'x_spend_p2',[10,25,50,100,200,400,800],'hp2_'),
    hinges(dfm,'recency',[7,14,28,56,112],'hr_'),
    hinges(dfm,'x_spend_12',[25,100,300],'hs12_'),
    hinges(dfm,'baskets_28',[1,3,6,10],'hb28_'),
    hinges(dfm,'x_bv_4',[10,25,50,100],'hbv_'),
    hinges(dfm,'days_28',[1,4,8,14],'hd28_'),
],axis=1)
def onehot_bins(frame, col, edges, prefix):
    x=frame[col].fillna(0).clip(lower=0).values.astype(float)
    b=np.digitize(x,edges)
    D=pd.get_dummies(b, prefix=prefix).astype(float); D.index=frame.index
    return D
OH = onehot_bins(dfm,'x_exp4w',[0,10,25,50,75,100,150,200,300,450,600],'ohx')
dfm2=pd.concat([dfm,H,OH],axis=1)
Hc=list(H.columns); OHc=list(OH.columns)
ridge_eval(dfm2,cols0,"ref")
ridge_eval(dfm2,cols0+Hc,"A h")
ridge_eval(dfm2,cols0+Hc+OHc,"B h+oh")
ridge_eval(dfm2,cols0+Hc+OHc+dcols,"C h+oh+dept")
ridge_eval(dfm2,cols0+Hc+OHc+dcols+scols,"D h+oh+dept+struct",verbose=True)
ridge_eval(dfm2,cols0+scols,"E struct only")
ridge_eval(dfm2,cols0+dcols+scols,"F dept+struct")
