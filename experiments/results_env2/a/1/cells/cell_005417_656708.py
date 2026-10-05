import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time
F = A.load_saved('allF.parquet'); W = A.load_saved('f_weekly.parquet')
ycol='future_spend_4w'
M = F.merge(W, on=['household_key','snapshot_day'], how='left')
base_cols = [c for c in F.columns if c not in ('household_key','snapshot_day',ycol)]
new_week = ['wk_1','wk_2','wk_3','wk_4','wk_5','wk_6','wk_7','wk_8','wk_13','wk_26','wk_39','wk_52',
            'wk_active_8','dow_0','dow_1','dow_2','dow_3','dow_4','dow_5','dow_6','wkend_share',
            'topstore_share_84','ncommod_84']
y = M[ycol].values.astype(float); d = M.snapshot_day.values

def cv(cols_list, eval_days=(403,431), seeds=(7,17), rounds=1200, lr=0.03, depth=6, decay=140.0):
    X = M[cols_list].values.astype(np.float32)
    out={}
    for ed in eval_days:
        trm = (d < ed) & ~np.isnan(y)
        tem = (d == ed) & ~np.isnan(y)
        w = 0.5**((ed - d[trm])/decay)
        ps = np.zeros(tem.sum())
        for s in seeds:
            m = xgb.XGBRegressor(n_estimators=rounds, learning_rate=lr, max_depth=depth,
                objective='reg:quantileerror', quantile_alpha=0.5, tree_method='hist',
                subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=-1)
            m.fit(X[trm], y[trm], sample_weight=w)
            ps += m.predict(X[tem])/len(seeds)
        out[ed]=round(np.abs(ps - y[tem]).mean(),3)
    return out

t0=time.time()
r1 = cv(base_cols); print('BASE', r1, round(time.time()-t0,1))
r2 = cv(base_cols+new_week); print('BASE+WEEK', r2, round(time.time()-t0,1))
