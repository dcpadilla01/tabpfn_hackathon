import agent_api as api
import pandas as pd, numpy as np
t = api.load_saved('cand_new.parquet')
e9 = api.load_saved('e009_macro.parquet')
m = t.merge(e9.drop(columns=['spend_28']), on=['household_key','snapshot_day'], how='inner')
tt = api.train_targets()
mm = m.merge(tt, on=['household_key','snapshot_day'])
y = mm['future_spend_4w'].values.astype(float)
cand = [c for c in t.columns if c not in ('household_key','snapshot_day')]
e9c = [c for c in e9.columns if c not in ('household_key','snapshot_day','spend_28')]
X = mm[e9c].copy()
num = X.select_dtypes(include=[np.number]).columns
X[num] = X[num].fillna(X[num].median())
cats = [c for c in e9c if c not in num]
Xn = X[num].values.astype(float)
mu, sd = Xn.mean(0), Xn.std(0)+1e-9
Xs = (Xn-mu)/sd
D = [Xs]
for c in cats:
    D.append(pd.get_dummies(mm[c].astype('category'), dummy_na=True).values.astype(float))
Xd = np.hstack(D)
lam = 10.0
w = np.linalg.solve(Xd.T@Xd + lam*np.eye(Xd.shape[1]), Xd.T@y)
res = y - Xd@w
print('train ridge MAE %.3f R2 %.3f'%(np.abs(res).mean(), 1-(res**2).sum()/((y-y.mean())**2).sum()))
cn = mm[cand].copy()
numc = cn.select_dtypes(include=[np.number]).columns
cn[numc] = cn[numc].fillna(cn[numc].median())
resn = []
for c in numc:
    v = cn[c].values.astype(float)
    ok = np.isfinite(v)
    if ok.sum()<100: continue
    r = np.corrcoef(v[ok], res[ok])[0,1]
    resn.append((c, r))
resn.sort(key=lambda z: -abs(z[1]))
print('top cand feats correlated with e009 residual (train):')
for c,r in resn[:20]: print(f'{c:20s} {r:+.3f}')