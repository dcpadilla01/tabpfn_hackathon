import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
T = agent_api.load_saved('e012_style.parquet')
tt = agent_api.train_targets()
df = T.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df.future_spend_4w.values.astype(float)
d = df.snapshot_day.values.astype(int)

num = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w') and pd.api.types.is_numeric_dtype(df[c])]
X = df[num].astype(float).copy()
# fill NaN with train medians
tr_all = d<=403
med = X[tr_all].median()
X = X.fillna(med).values.astype(float)
# scale-adjust the window sums
for c in ['spend_84','spend_112']: X[:, num.index(c)]/=3.0
for c in ['spend_182']: X[:, num.index(c)]/=6.5
for c in ['spend_365','total_all']: X[:, num.index(c)]/=13.0
for c in ['blk_1','blk_2','blk_3','blk_4','blk_5','blk_6','blk_7','blk_8','blk_9','blk_10','blk_11','blk_12','blk_13','spend_s336','spend_s364','spend_s392','max6','d_block_max']:
    X[:, num.index(c)] = X[:, num.index(c)]  # blocks are 28d sums, fine

X1 = np.hstack([X, np.ones((len(X),1))])
def ridge_fit(Xtr, ytr, alpha):
    mu, sd = Xtr[:,:-1].mean(0), Xtr[:,:-1].std(0); sd[sd==0]=1
    Z = np.hstack([(Xtr[:,:-1]-mu)/sd, np.ones((len(Xtr),1))])
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[-1,-1]-=alpha
    return (mu, sd, np.linalg.solve(A, Z.T@ytr))
def ridge_pred(Xte, m):
    mu, sd, w = m
    return np.hstack([(Xte[:,:-1]-mu)/sd, np.ones((len(Xte),1))])@w

tr, te = d<=403, d==431
for alpha in [3,30,300,3000]:
    m = ridge_fit(X1[tr], y[tr], alpha)
    print(f"ridge-all alpha={alpha}: holdout(431) MAE {np.mean(np.abs(ridge_pred(X1[te],m)-y[te])):.2f}")

# heavy tail: within top decile, what correlates with residual?
pred = 0.5*df.spend_84.values/3+0.3*df.spend_28.values+0.2*df.avg28_all.values
resid = y - pred
heavy = y > 366
rows=[]
for c in num:
    a = df[c].astype(float).values
    m = np.isfinite(a) & heavy
    if m.sum()<500: continue
    med2 = np.nanmedian(a[heavy])
    aa = np.where(np.isfinite(a), a, med2)
    if aa[heavy].std()==0: continue
    r = np.corrcoef(aa[heavy], resid[heavy])[0,1]
    rows.append((abs(r), r, c))
rows.sort(reverse=True)
print("\nHeavy (y>366, n=%d) residual corr:" % heavy.sum())
for _,r,c in rows[:20]: print(f"  {r:+.3f} {c}")
