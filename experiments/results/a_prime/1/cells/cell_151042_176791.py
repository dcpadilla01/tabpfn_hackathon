import agent_api as api
import pandas as pd, numpy as np

t = api.train_targets()
y = t.future_spend_4w.values.astype(float)
print("targets:", t.shape, "zero share %.3f" % (y==0).mean())
print(t.future_spend_4w.describe(percentiles=[.25,.5,.75,.9,.95,.99]).round(1))
zz = pd.Series(y==0).groupby(t.snapshot_day).mean().round(3)
gm = t.groupby('snapshot_day').future_spend_4w.agg(['mean','median']).round(1)
gm['zero']=zz
print(gm)

tables = {
 'E000': api.baseline_features(),
 'E001': api.load_saved('e001_txhist.parquet'),
 'E002': api.load_saved('e002_channel.parquet'),
 'E003': api.load_saved('e003_catmix.parquet'),
 'E004': api.load_saved('e004_mkt.parquet'),
 'E006': api.load_saved('e006_catmix_mkt.parquet'),
 'E007': api.load_saved('e007_logratio.parquet'),
}
for k,v in tables.items(): print(k, v.shape)
print('E001 cols:', ','.join(map(str,tables['E001'].columns)))
e3x = [c for c in tables['E003'].columns if c not in tables['E001'].columns]
print('E003 extra:', ','.join(map(str,e3x)))

KEY=['household_key','snapshot_day']
def matrix(df):
    m = t.merge(df, on=KEY, how='left')
    yv = m.future_spend_4w.values.astype(float)
    d = m.snapshot_day.values
    X = m.drop(columns=KEY+['future_spend_4w'])
    cols=[]
    for c in X.columns:
        s=X[c]
        if s.dtype==object or str(s.dtype).startswith('category') or s.dtype==bool:
            v=pd.factorize(s)[0].astype(float)
        else:
            v=pd.to_numeric(s,errors='coerce').values.astype(float)
        cols.append(v)
    Xm = np.column_stack(cols) if cols else np.zeros((len(yv),0))
    return Xm, yv, d, list(X.columns)

def ridge_eval(X, y, days, lams=(0.01,0.1,1,10,100)):
    fit = days<=347; val = days>=375
    Xf=X[fit]
    mu=np.nanmean(Xf,0); sd=np.nanstd(Xf,0); sd[sd<1e-9]=1
    Z=np.where(np.isfinite(X),(X-mu)/sd,0.0)
    Zf=np.hstack([Z[fit],np.ones((fit.sum(),1))]); Zv=np.hstack([Z[val],np.ones((val.sum(),1))])
    p=Zf.shape[1]; A=Zf.T@Zf; b=Zf.T@y[fit]
    out=[]
    for lam in lams:
        w=np.linalg.solve(A+lam*np.eye(p),b)
        out.append((np.abs(Zv@w-y[val]).mean(),lam))
    out.sort(); return out[0]

med = np.median(y[(t.snapshot_day<=347).values])
print('baseline fit-median -> offline MAE %.2f' % np.abs(med - y[(t.snapshot_day>=375).values]).mean())

for k in ['E000','E001','E002','E003','E004','E006','E007']:
    X,yv,d,cols = matrix(tables[k])
    mae,lam = ridge_eval(X,yv,d)
    print(k, X.shape, 'ridge offline MAE %.2f (lam %g)'%(mae,lam))

# univariate screen on E003 (contains E001 features too)
X,yv,d,cols = matrix(tables['E003'])
res=[]
for j,c in enumerate(cols):
    mae,lam = ridge_eval(X[:,[j]], yv, d)
    res.append((mae,c))
res.sort()
print('Top univariate (E003):')
for mae,c in res[:18]: print('  %.2f %s'%(mae,c))