import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='left')
for c in feats.columns:
    if feats[c].dtype.kind not in 'fi':
        df[c] = df[c].astype('category').cat.codes
feat_cols = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = df[feat_cols].astype(float).values
y = df['future_spend_4w'].values
tr = np.where(df.snapshot_day <= 403)[0]; te = np.where(df.snapshot_day == 431)[0]
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
def mae(p): return round(np.abs(p-yte).mean(),3)
def xq(Xtr,ytr,Xte,alpha=0.5,seed=7,n=400):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=5,
        min_child_weight=40, learning_rate=0.08, n_estimators=n, subsample=0.9,
        colsample_bytree=0.8, n_jobs=8, tree_method='hist', random_state=seed)
    m.fit(Xtr,ytr); return m.predict(Xte)
pb = np.mean([xq(Xtr,ytr,Xte,seed=s) for s in (1,2,3,4,5)], axis=0)
print('bag:', mae(pb))
# affine calibration on internal test
from scipy.optimize import minimize
def obj(th): return np.abs(th[0]*pb+th[1]-yte).mean()
r = minimize(obj, [1.0,0.0], method='Nelder-Mead')
print('affine calib a,b:', r.x.round(4), 'mae:', round(r.fun,3))
# median bias
print('pred median', np.median(pb), 'target median', np.median(yte), 'pred mean', pb.mean(), 'tgt mean', yte.mean())
# MAE-optimal per-decile shift: bin by prediction, find best additive shift per bin
qs = np.quantile(pb, np.linspace(0,1,11))
bins = np.clip(np.searchsorted(qs, pb, side='right')-1, 0, 9)
p2 = pb.copy()
for b in range(10):
    m = bins==b
    if m.sum()>50:
        cand = np.median(yte[m])-np.median(pb[m])
        p2[m] = pb[m]+cand
print('per-decile median-shift:', mae(p2))
# seasonal feature test: spend in 28d window ending at s-364 and s-336
tr_ = agent_api.snapshot(459).table('transactions')
def seasonal(snap_day, hh_set, offsets=(336,364,392)):
    out = {}
    t = tr_[tr_.day <= snap_day]
    g = t.groupby(['household_key','day']).sales_value.sum().reset_index()
    for off in offsets:
        lo, hi = snap_day-off-27, snap_day-off
        m = (g.day>=lo)&(g.day<=hi)
        s = g[m].groupby('household_key').sales_value.sum()
        out[f'seas_{off}'] = s
    return out
# build for snapshot 431 internal test rows
hh_te = df.household_key.values[te]; sd = 431
seas = seasonal(sd, None)
Xseas = np.zeros((len(te),3))
for i,off in enumerate((336,364,392)):
    Xseas[:,i] = hh_te.map(seas[f'seas_{off}']).fillna(0).values if hasattr(hh_te,'map') else pd.Series(hh_te).map(seas[f'seas_{off}']).fillna(0).values
for i,off in enumerate((336,364,392)):
    print(f'corr seas_{off} vs yte:', round(np.corrcoef(Xseas[:,i], yte)[0,1],3), 'MAE alone:', mae(Xseas[:,i]))
Xs2 = np.hstack([Xte, Xseas])
Xs2tr = np.hstack([Xtr, np.zeros((len(tr),3))])
p_s = xq(Xs2tr, ytr, Xs2); print('with seasonal feats (zeros in train):', mae(p_s))
