import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
tt = agent_api.train_targets()
base = agent_api.load_saved("e013_stock.parquet")
keys=["household_key","snapshot_day"]
df = tt.merge(base,on=keys,how="left")
cols=[c for c in base.columns if c not in keys]
for c in cols:
    if df[c].dtype=='object' or str(df[c].dtype)=='category':
        df[c]=df[c].astype('object').astype('category').cat.codes.astype(float).replace(-1,np.nan)
fit=df[df.snapshot_day<=403]; hold=df[df.snapshot_day==431]
def prep(Xf,Xh):
    med=np.nanmedian(Xf,axis=0); Xf=np.where(np.isnan(Xf),med,Xf); Xh=np.where(np.isnan(Xh),med,Xh)
    mu=Xf.mean(0); sd=Xf.std(0)+1e-9
    return (Xf-mu)/sd,(Xh-mu)/sd
Xf, Xh = prep(fit[cols].astype(float).values, hold[cols].astype(float).values)
Xf=np.hstack([Xf,np.ones((len(Xf),1))]); Xh=np.hstack([Xh,np.ones((len(Xh),1))])
yf=fit.future_spend_4w.values; yh=hold.future_spend_4w.values
A=Xf[fit.snapshot_day<=375]; w=np.linalg.solve(A.T@A+32*np.eye(A.shape[1]),A.T@yf[fit.snapshot_day<=375])
ph=np.clip(Xh@w,0,None)
res=ph-yh
yh0=yh==0
print("HOLD 431: n=%d  MAE=%.2f"%(len(yh),np.abs(res).mean()))
print("zeros: n=%d (%.0f%%)  mean|res|=%.1f  mean pred on zeros=%.1f"%(yh0.sum(),100*yh0.mean(),np.abs(res[yh0]).mean(),ph[yh0].mean()))
print("actives: mean|res|=%.1f"%np.abs(res[~yh0]).mean())
for lo,hi in [(0,50),(50,150),(150,300),(300,600),(600,3000)]:
    m=(yh>=lo)&(yh<hi)
    print(f"y in [{lo},{hi}): n={m.sum():4d} mean pred={ph[m].mean():7.1f} mean y={yh[m].mean():7.1f} MAE={np.abs(res[m]).mean():6.1f}")
# deciles of prediction
q=np.quantile(ph,[.5,.75,.9,.95,.99])
print("pred quantiles:",q.round(1))
print("y quantiles:",np.quantile(yh,[.5,.75,.9,.95,.99]).round(1))
# how much would perfect zero-classification help: set pred=0 where yh==0
print("MAE if zeros perfectly identified:", np.abs(np.where(yh0,0,ph)-yh).mean())
# error on train fit rows (in-sample-ish) by y bucket
ptr=np.clip(Xf@w,0,None); rtr=ptr-yf
for lo,hi in [(0,1),(1,50),(50,150),(150,300),(300,600),(600,3000)]:
    m=(yf>=lo)&(yf<hi)
    print(f"TRAIN y in [{lo},{hi}): n={m.sum():5d} mean pred={ptr[m].mean():7.1f} mean y={yf[m].mean():7.1f} MAE={np.abs(rtr[m]).mean():6.1f}")
print("TRAIN zeros: n=%d mean pred=%.1f MAE=%.1f"%((yf==0).sum(),ptr[yf==0].mean(),np.abs(rtr[yf==0]).mean()))
