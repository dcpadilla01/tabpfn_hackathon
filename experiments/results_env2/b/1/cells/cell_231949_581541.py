import agent_api, numpy as np, pandas as pd
t = agent_api.load_saved('e008_level_shape.parquet')
print('E008 shape', t.shape)
feat = [c for c in t.columns if c not in ('household_key','snapshot_day')]
print('n_feat', len(feat))
print('dtypes', t[feat].dtypes.value_counts().to_dict())
print('cols:', feat)
tt = agent_api.train_targets()
print('train_targets', tt.shape)
print({k: round(v,2) for k,v in tt.future_spend_4w.describe().to_dict().items()})
m = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', m.shape, 'snapdays', sorted(m.snapshot_day.unique()))
y = m.future_spend_4w.values.astype(float)
print('zero frac', round((y==0).mean(),3))
Xdf = m[feat].copy()
for c in feat:
    dt = str(Xdf[c].dtype)
    if dt in ('object','category','bool'):
        Xdf[c] = Xdf[c].astype('category').cat.codes.replace(-1, np.nan)
X = Xdf.values.astype(float)
print('X', X.shape, 'nanfrac', round(np.isnan(X).mean(),4))

def prep(Xtr, Xev):
    med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0.0, med)
    A = np.where(np.isnan(Xtr), med, Xtr); B = np.where(np.isnan(Xev), med, Xev)
    mu = A.mean(0); sd = A.std(0); sd[sd<1e-12]=1
    return (A-mu)/sd, (B-mu)/sd

def ridge_fit(Z, yy, a):
    p = Z.shape[1]
    return np.linalg.solve(Z.T@Z + a*np.eye(p), Z.T@yy)

folds = [375, 403, 431]
for logt in [False, True]:
    for a in [3,10,30,100,300]:
        maes=[]
        for e in folds:
            tr = m.snapshot_day.values < e; ev = m.snapshot_day.values == e
            Ztr, Zev = prep(X[tr], X[ev])
            yy = np.log1p(y) if logt else y
            w = ridge_fit(Ztr, yy[tr], a)
            pr = Zev@w
            if logt: pr = np.expm1(np.clip(pr, 0, 12))
            maes.append(np.abs(pr - y[ev]).mean())
        print('logt' if logt else 'raw ', 'a', a, 'MAE', np.round(maes,2), 'avg', round(np.mean(maes),3))
for e in folds:
    tr = m.snapshot_day.values < e; ev = m.snapshot_day.values == e
    print('meanbase', e, round(np.abs(y[tr].mean()-y[ev]).mean(),3))
Xf = np.where(np.isnan(X), np.nanmedian(X,0), X)
cs = np.array([abs(np.corrcoef(Xf[:,j], y)[0,1]) if np.std(Xf[:,j])>0 else 0 for j in range(X.shape[1])])
order = np.argsort(-cs)
print('top25 |corr|:')
for j in order[:25]: print('  ', feat[j], round(cs[j],3))
for K in [20,40,60,80,120,163]:
    sub = order[:K]
    maes=[]
    for e in folds:
        tr = m.snapshot_day.values < e; ev = m.snapshot_day.values == e
        Ztr, Zev = prep(X[tr][:,sub], X[ev][:,sub])
        w = ridge_fit(Ztr, y[tr], 30)
        maes.append(np.abs(Zev@w - y[ev]).mean())
    print('topK', K, 'ridge a30 raw MAE', round(np.mean(maes),3))
