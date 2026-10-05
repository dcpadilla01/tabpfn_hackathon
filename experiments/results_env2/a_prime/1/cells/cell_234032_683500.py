import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
T = agent_api.load_saved('e012_style.parquet')
tt = agent_api.train_targets()
df = T.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df.future_spend_4w.values.astype(float)
d = df.snapshot_day.values.astype(int)
num = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w') and pd.api.types.is_numeric_dtype(df[c])]
X = df[num].astype(float).copy()
print("inf counts:", np.isinf(X.values).sum())
X = X.replace([np.inf,-np.inf], np.nan)
tr_all = d<=403
med = X[tr_all].median()
X = X.fillna(med).values.astype(float)
for c,i in [('spend_84',num.index('spend_84')),('spend_112',num.index('spend_112'))]: X[:,i]/=3.0
X[:,num.index('spend_182')]/=6.5
for c in ['spend_365','total_all']: X[:,num.index(c)]/=13.0

def ridge_fit(Xtr, ytr, alpha):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
    Z = np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))])
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[-1,-1]-=alpha
    return (mu, sd, np.linalg.solve(A, Z.T@ytr))
def ridge_pred(Xte, m):
    mu, sd, w = m
    return np.hstack([(Xte-mu)/sd, np.ones((len(Xte),1))])@w
tr, te = d<=403, d==431
for alpha in [3,30,300,3000]:
    m = ridge_fit(X[tr], y[tr], alpha)
    print(f"ridge-all alpha={alpha}: holdout(431) MAE {np.mean(np.abs(ridge_pred(X[te],m)-y[te])):.2f}")
m = ridge_fit(X[tr_all], y[tr_all], 30)
pv = ridge_pred(X[d>=459], m)
print("val preds: mean %.1f, neg %d" % (pv.mean(), (pv<0).sum()))

# heavy segment: regress residual on features (ridge, small alpha)
pred = 0.5*df.spend_84.values/3+0.3*df.spend_28.values+0.2*df.avg28_all.values
resid = y - pred
heavy = (y>366) & tr_all
Xh, rh = X[heavy], resid[heavy]
mh = ridge_fit(Xh, rh, 300)
ph = ridge_pred(Xh, mh)
print("\nheavy resid ridge in-sample R2:", 1 - np.var(rh-ph)/np.var(rh))
# top positive weights
w = mh[2][:-1]
order = np.argsort(-w)[:15]
print("top positive weights (heavy resid):")
for i in order: print(f"  {w[i]:+.2f} {num[i]}")
