import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time
t0=time.time()
F = A.load_saved('allF.parquet'); W = A.load_saved('f_weekly.parquet')
ycol='future_spend_4w'
M = F.merge(W, on=['household_key','snapshot_day'], how='left')
new_week = ['wk_1','wk_2','wk_3','wk_4','wk_5','wk_6','wk_7','wk_8','wk_13','wk_26','wk_39','wk_52',
            'wk_active_8','dow_0','dow_1','dow_2','dow_3','dow_4','dow_5','dow_6','wkend_share',
            'topstore_share_84','ncommod_84']
tr_days=[95,123,151,179,207,235,263,291,319,347,375,403,431]

def feats(df, extra):
    cols = [c for c in df.columns if c not in ('household_key','snapshot_day',ycol)]
    if extra is not None:
        cols = cols + [c for c in extra if c not in cols]
    return df[cols].astype(float)

def run(Xdf, y, days, w_ref, eval_days, seeds=(7,17,27), lr=0.03, rounds=2400, depth=6, decay=140.0, subs=0.8, cols=0.8):
    maes=[]; preds={}
    for ed in eval_days:
        trm = days < ed
        Xtr, ytr = Xdf[trm], y[trm]
        w = 0.5**((w_ref - days[trm])/decay)
        ps = np.zeros((~trm).sum(), dtype=float)
        for s in seeds:
            m = xgb.XGBRegressor(n_estimators=rounds, learning_rate=lr, max_depth=depth,
                objective='reg:quantileerror', quantile_alpha=0.5, tree_method='hist',
                subsample=subs, colsample_bytree=cols, random_state=s, n_jobs=32)
            m.fit(Xtr, ytr, sample_weight=w)
            ps += m.predict(Xdf[~trm])/len(seeds)
        yy = y[~trm]
        maes.append(np.abs(ps-yy).mean()); preds[ed]=ps
    return maes, preds

base_cols = [c for c in F.columns if c not in ('household_key','snapshot_day',ycol)]
for name, extra in [('BASE', None), ('BASE+WEEK', new_week)]:
    X = feats(M, extra).values.astype(np.float32)
    y = M[ycol].values.astype(float); d = M.snapshot_day.values
    maes,_ = run(X, y, d, 431, [403,431])
    print(name, 'MAE@403,431:', [round(m,3) for m in maes], 'time', round(time.time()-t0,1))
