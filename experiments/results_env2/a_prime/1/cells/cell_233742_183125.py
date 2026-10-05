import numpy as np, pandas as pd
T = agent_api.load_saved('e012_style.parquet')
tt = agent_api.train_targets()
df = T.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df.future_spend_4w.values.astype(float)
def mae(p): return float(np.mean(np.abs(np.asarray(p)-y)))

blk = df[[f'blk_{i}' for i in range(1,14)]].values.astype(float)
med_blk = np.nanmedian(blk, axis=1)
trim_blk = np.where(np.isnan(blk), 0, blk)
top = np.nanmax(np.where(np.isnan(blk), -1, blk), axis=1)
trim_sum = np.nansum(np.where(blk==top[:,None], 0, np.where(np.isnan(blk),0,blk)), axis=1)
nobs = np.sum(~np.isnan(blk), axis=1)
trim_blk_mean = np.where(nobs>1, trim_sum/np.maximum(nobs-1,1), med_blk)

cands = {
 'global_mean': np.full(len(y), y.mean()),
 'spend_28': df.spend_28.values,
 '0.9*spend_28': 0.9*df.spend_28.values,
 'spend_84': df.spend_84.values,
 'avg28_all': df.avg28_all.values,
 'ewma_hl28': df.d_ewma_spend_hl28.values,
 'med_blk': med_blk,
 'trim_blk': trim_blk_mean,
 'blend .5s84+.3s28+.2avg': 0.5*df.spend_84.values+0.3*df.spend_28.values+0.2*df.avg28_all.values,
 'blend .4s84+.3s28+.3avg': 0.4*df.spend_84.values+0.3*df.spend_28.values+0.3*df.avg28_all.values,
 'max(s28, .8*s84)': np.maximum(df.spend_28.values, 0.8*df.spend_84.values),
}
for k,v in cands.items(): print(f"{k:28s} MAE {mae(v):8.2f}")

# ridge on a few top features, day-holdout (fit <=403, predict 431)
feats = ['spend_28','spend_84','spend_182','spend_365','avg28_all','d_ewma_spend_hl28','d_ewma_spend_hl112',
         'trips_28','trips_84','recency','tenure','blk_1','d_wk_spend_mean_12w','basket_mean_84','zero6','decay_mean']
X = df[feats].fillna(0).values.astype(float)
def ridge_fit(Xtr, ytr, alpha):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
    Z = (Xtr-mu)/sd
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[-1,-1]-=alpha  # no penalty on intercept col
    Z1 = np.hstack([Z, np.ones((len(Z),1))])
    w = np.linalg.solve(A, Z1.T@ytr)
    return mu, sd, w
def ridge_pred(Xte, m):
    mu, sd, w = m
    Z = (Xte-mu)/sd
    return np.hstack([Z, np.ones((len(Z),1))])@w
d = df.snapshot_day.values.astype(int)
tr, te = d<=403, d==431
best=None
for alpha in [1,10,100,1000]:
    m = ridge_fit(X[tr], y[tr], alpha)
    p = ridge_pred(X[te], m)
    mm = mae(p[te]) if False else float(np.mean(np.abs(p-y[te])))
    print(f"ridge alpha={alpha}: holdout(431) MAE {mm:.2f}")
m = ridge_fit(X, y, 100)
print("ridge alpha=100 in-sample MAE:", float(np.mean(np.abs(ridge_pred(X,m)-y))))

# probe: load_saved inside build_features
def probe(view, snapshot_day):
    e = agent_api.load_saved('e012_style.parquet')
    e = e[e.snapshot_day==snapshot_day]
    out = e.set_index('household_key').drop(columns=['snapshot_day'])
    out['probe_ok'] = 1.0
    return out
res = agent_api.build_features(probe)
print("probe result:", res.shape, "probe_ok sum:", res.probe_ok.sum(), "days:", sorted(res.snapshot_day.unique()))
