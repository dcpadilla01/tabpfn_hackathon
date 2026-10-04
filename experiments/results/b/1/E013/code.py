import agent_api as api
names = ['e012_full','e009_macro','e008_decomp2','e004_long_hist','e006_seq_gaps','e011_display']
for nm in names:
    t = api.load_saved(nm + '.parquet')
    print('===', nm, t.shape)
    print([c for c in t.columns if c not in ('household_key','snapshot_day')])
    print()

# ---- cell ----
import agent_api as api
t = api.load_saved('cand_new.parquet')
print(t.shape)
cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
print(len(cols), cols)
# check overlap with e009
e9 = api.load_saved('e009_macro.parquet')
e9c = [c for c in e9.columns if c not in ('household_key','snapshot_day')]
print('overlap with e009:', set(cols)&set(e9c))
# quick: distribution of target-ish? just check a few cand features describe
print(t[['a2','a7','dsl','rvu','ly_spend']].describe().T)

# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np
t = api.load_saved('cand_new.parquet')
e9 = api.load_saved('e009_macro.parquet')
e8 = api.load_saved('e008_decomp2.parquet')
m = t.merge(e9.drop(columns=['spend_28']), on=['household_key','snapshot_day'], how='inner')
print('merged', m.shape)
# how many candidate features add on top of e009? quick correlation of each cand feat with residual proxy?
# We can't see val targets. But train targets available:
tt = api.train_targets()
print(tt.shape, tt.head())
mm = m.merge(tt, on=['household_key','snapshot_day'])
y = mm['future_spend_4w']
cand = [c for c in t.columns if c not in ('household_key','snapshot_day','spend_28')]
e9c = [c for c in e9.columns if c not in ('household_key','snapshot_day')]
# fit ridge on e9 on train rows, get residuals, then corr of cand feats with residual
import numpy as np
X = mm[e9c].copy()
num = X.select_dtypes(include=[np.number]).columns
X[num] = X[num].fillna(X[num].median())
# simple: use HistGradientBoosting? not allowed import sklearn. Use numpy ridge.
Xn = X[num].values.astype(float)
# standardize
mu, sd = Xn.mean(0), Xn.std(0)+1e-9
Xs = (Xn-mu)/sd
yv = y.values.astype(float)
# one-hot categoricals
cats = [c for c in e9c if c not in num]
print('cat cols in e9:', cats)
D = [Xs]
for c in cats:
    d = pd.get_dummies(mm[c].astype('category'), dummy_na=True).values.astype(float)
    D.append(d)
Xd = np.hstack(D)
lam = 10.0
A = Xd.T@Xd + lam*np.eye(Xd.shape[1])
b = Xd.T@yv
w = np.linalg.solve(A,b)
pred = Xd@w
res = yv - pred
print('train ridge MAE', np.abs(res).mean(), 'R2', 1-((res**2).sum()/((yv-yv.mean())**2).sum()))
# now corr of each cand feature with residual
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
for c,r in resn[:20]: print(f'{c:20s} {r:+.3f}')

# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np
t = api.load_saved('cand_new.parquet')
e9 = api.load_saved('e009_macro.parquet')
m = t.merge(e9.drop(columns=['spend_28']), on=['household_key','snapshot_day'], how='inner')
tt = api.train_targets()
mm = m.merge(tt, on=['household_key','snapshot_day'])
y = mm['future_spend_4w'].values.astype(float)
cand = [c for c in t.columns if c not in ('household_key','snapshot_day')]
e9c = [c for c in e9.columns if c not in ('household_key','snapshot_day')]
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

# ---- cell ----
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

# ---- cell ----
import agent_api as api
import pandas as pd
t = api.load_saved('cand_new.parquet')
e9 = api.load_saved('e009_macro.parquet')
cand = [c for c in t.columns if c not in ('household_key','snapshot_day')]
e9c = [c for c in e9.columns if c not in ('household_key','snapshot_day')]
m = t.merge(e9.drop(columns=['spend_28']), on=['household_key','snapshot_day'], how='inner')
print(m.shape, len(cand)+len(e9c))
p = api.save_table(m, 'e013_union')
print(p)