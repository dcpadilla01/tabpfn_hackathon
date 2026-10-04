import agent_api, pandas as pd, numpy as np

tt = agent_api.train_targets()
mkt = agent_api.load_saved('mkt_v2.parquet')
df = mkt.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in mkt.columns if c not in ('household_key','snapshot_day')]
all_train_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]

def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0)+1e-9
    Xs = (X-mu)/sd
    A = Xs.T@Xs + alpha*np.eye(Xs.shape[1])
    w = np.linalg.solve(A, Xs.T@y)
    return w, mu, sd

def ridge_pred(X, w, mu, sd):
    return np.clip((X-mu)/sd @ w, 0, None)

def eval_split(fit_days, val_days, transform=None, alpha=100.0):
    tr = df[df.snapshot_day.isin(fit_days)]
    va = df[df.snapshot_day.isin(val_days)]
    Xtr, ytr = tr[feats].fillna(0).values, tr.future_spend_4w.values
    Xva, yva = va[feats].fillna(0).values, va.future_spend_4w.values
    if transform == 'log':
        w,mu,sd = ridge_fit(Xtr, np.log1p(ytr), alpha)
        pv = np.expm1(np.clip((Xva-mu)/sd @ w, 0, 20))
    else:
        w,mu,sd = ridge_fit(Xtr, ytr, alpha)
        pv = ridge_pred(Xva, w, mu, sd)
    return np.abs(pv-yva).mean()

print("== internal ridge CV on mkt_v2 (72 feats) ==")
for alpha in [10,100,1000]:
    print(f"alpha={alpha}: val431={eval_split([d for d in all_train_days if d!=431],[431],alpha=alpha):.3f}, val403={eval_split([d for d in all_train_days if d!=403],[403],alpha=alpha):.3f}")
print("log-target: val431=", round(eval_split([d for d in all_train_days if d!=431],[431],'log',10),3),
      " val403=", round(eval_split([d for d in all_train_days if d!=403],[403],'log',10),3))

# naive baselines
for col in ['spend_84','spend_28','x_exp4w']:
    for vd in [431,403]:
        va = df[df.snapshot_day==vd]
        print(f"naive {col} -> {vd}: {np.abs(va[col].fillna(0).values - va.future_spend_4w.values).mean():.3f}")

# recency-bucket conditional median lookup (uses targets: analysis only)
def recency_lookup(fit_days, val_days):
    tr = df[df.snapshot_day.isin(fit_days)].copy()
    va = df[df.snapshot_day==val_days].copy()
    bins = [-1,7,14,21,28,42,56,84,112,10000]
    tr['rb'] = pd.cut(tr.recency, bins)
    lut = tr.groupby('rb', observed=True).future_spend_4w.median()
    va['rb'] = pd.cut(va.recency, bins)
    pv = va['rb'].map(lut).fillna(0).values
    return np.abs(pv - va.future_spend_4w.values).mean(), lut
for vd in [431,403]:
    fd = [d for d in all_train_days if d!=vd]
    m,lut = recency_lookup(fd, vd)
    print(f"recency-median lookup val{vd}: {m:.3f}")
print(lut)
