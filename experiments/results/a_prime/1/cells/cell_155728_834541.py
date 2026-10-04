import numpy as np, pandas as pd, agent_api as api
t = api.load_saved('e003_catmix.parquet')
tt = api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='left')
feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
tr = df.snapshot_day <= 431; va = df.snapshot_day >= 459
Xtr = df.loc[tr, feats].astype(float).values
mu = np.nanmean(Xtr, 0); Xtr = np.where(np.isnan(Xtr), mu, Xtr)
ytr = df.loc[tr,'future_spend_4w'].values
Xva = df.loc[va, feats].astype(float).values; Xva = np.where(np.isnan(Xva), mu, Xva)
yva = df.loc[va,'future_spend_4w'].values
sd = Xtr.std(0); sd[sd==0]=1
Ztr=(Xtr-mu)/sd; Zva=(Xva-mu)/sd
def fit(lam):
    return np.linalg.solve(Ztr.T@Ztr + lam*np.eye(len(feats)), Ztr.T@ytr)
print('median-only MAE %.3f' % np.abs(np.median(ytr)-yva).mean())
best=None
for lam in [3,10,30,100,300,1000]:
    pv = Zva@fit(lam)
    m = np.abs(pv-yva).mean()
    print('ridge lam %g MAE %.3f' % (lam, m))
w = fit(100); pv = Zva@w; res = yva - pv
dva = df.loc[va].copy(); dva['pred']=pv; dva['res']=res; dva['y']=yva
print('\nBy true-y bucket:')
dva['yb'] = pd.cut(dva.y, [-1,0.5,50,150,300,700,1e9])
print(dva.groupby('yb', observed=True).apply(lambda g: pd.Series({'n':len(g),'mae':np.abs(g.res).mean(),'mean_pred':g.pred.mean(),'mean_y':g.y.mean()})))
print('\nBy tenure bucket:')
dva['tb'] = pd.cut(dva.tenure, [0,120,300,600,1e9])
print(dva.groupby('tb', observed=True).apply(lambda g: pd.Series({'n':len(g),'mae':np.abs(g.res).mean()})))
print('\nBy days_since_last:')
dva['db'] = pd.cut(dva.days_since_last, [-1,0,7,14,28,60,1e9])
print(dva.groupby('db', observed=True).apply(lambda g: pd.Series({'n':len(g),'mae':np.abs(g.res).mean(),'bias':g.res.mean()})))
print('\ncorr(pred,y)=%.3f' % np.corrcoef(pv,yva)[0,1])