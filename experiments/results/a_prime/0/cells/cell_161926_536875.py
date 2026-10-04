import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

tt = agent_api.train_targets()
y_all = tt.future_spend_4w
print("train rows", len(tt), "zero share %.3f" % (y_all==0).mean())
print(y_all.describe())
print("p50/75/90/95/99:", np.percentile(y_all,[50,75,90,95,99]).round(1))

for name in ["structure_v1.parquet","hist_v2.parquet","mix_v1.parquet","season_v1.parquet"]:
    df = agent_api.load_saved(name)
    print("==",name,df.shape); print(list(df.columns))

base = agent_api.load_saved("e013_stock.parquet")
keys=["household_key","snapshot_day"]
cols=[c for c in base.columns if c not in keys]
df = tt.merge(base,on=keys,how="left")
cat_cols=[c for c in cols if df[c].dtype=='object' or str(df[c].dtype)=='category']
print("cat cols:",cat_cols)
for c in cat_cols:
    df[c]=df[c].astype('object').astype('category')
    df[c]=df[c].cat.codes.astype(float).replace(-1,np.nan)

def ridge_eval(frame, feat_cols, tag, alphas=(0.5,1,2,4,8,16,32)):
    fit=frame[frame.snapshot_day<=403]; hold=frame[frame.snapshot_day==431]
    yf=fit.future_spend_4w.values; yh=hold.future_spend_4w.values
    Xf=fit[feat_cols].astype(float).values; Xh=hold[feat_cols].astype(float).values
    med=np.nanmedian(Xf,axis=0)
    Xf=np.where(np.isnan(Xf),med,Xf); Xh=np.where(np.isnan(Xh),med,Xh)
    mu=Xf.mean(0); sd=Xf.std(0)+1e-9
    Xf=(Xf-mu)/sd; Xh=(Xh-mu)/sd
    Xf=np.hstack([Xf,np.ones((len(Xf),1))]); Xh=np.hstack([Xh,np.ones((len(Xh),1))])
    itr=(fit.snapshot_day<=375).values; iva=(fit.snapshot_day==403).values
    best=None
    for a in alphas:
        A=Xf[itr]; G=A.T@A+a*np.eye(A.shape[1]); w=np.linalg.solve(G,A.T@yf[itr])
        mae=np.abs(Xf[iva]@w-yf[iva]).mean()
        if best is None or mae<best[1]: best=(a,mae)
    a=best[0]
    G=Xf.T@Xf+a*np.eye(Xf.shape[1]); w=np.linalg.solve(G,Xf.T@yf)
    ph=np.clip(Xh@w,0,None)
    mae=np.abs(ph-yh).mean()
    s28=hold['spend_28'].fillna(0).values
    print(f"[{tag}] alpha={a} innerMAE={best[1]:.2f} hold431 MAE={mae:.3f} | naive spend_28 MAE={np.abs(s28-yh).mean():.3f} corr={np.corrcoef(ph,yh)[0,1]:.3f}")
    return mae

m0=ridge_eval(df,cols,"base e013")
# correlation screen
fit=df[df.snapshot_day<=403]
cors={}
for c in cols:
    x=fit[c].astype(float).values; yy=fit.future_spend_4w.values
    ok=~np.isnan(x)
    if ok.sum()>100: cors[c]=np.corrcoef(x[ok],yy[ok])[0,1]
cs=sorted(cors.items(),key=lambda kv:-abs(kv[1]))
print("top |corr| with target:", [(k,round(v,3)) for k,v in cs[:25]])
print("bottom:", [(k,round(v,3)) for k,v in cs[-10:]])
