import agent_api as api, pandas as pd, numpy as np, time
import xgboost as xgb
allF = api.load_saved('allF.parquet')
DROP = ['household_key','snapshot_day','future_spend_4w']
FEATS = [c for c in allF.columns if c not in DROP]
tt = api.train_targets()[['household_key','snapshot_day','future_spend_4w']]
df = allF.drop(columns=['future_spend_4w']).merge(tt, on=['household_key','snapshot_day'])
tr = df[df.snapshot_day<=375]; te = df[df.snapshot_day.isin([403,431])]
print("tr:", tr.shape, "te:", te.shape, "n_feats:", len(FEATS))

def w(day, halflife=140, ref=375): return 0.5**((ref-day)/halflife)

def run(model_type, alpha=0.5, tweedie_p=1.3, lr=0.03, rounds=1200, seeds=(1,2), logt=False):
    t0=time.time(); preds=[]
    Xtr, ytr = tr[FEATS], tr['future_spend_4w']
    wt = w(tr['snapshot_day'].values)
    yt = np.log1p(ytr) if logt else ytr
    for s in seeds:
        if model_type=='quantile':
            m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, n_estimators=rounds, learning_rate=lr,
                                 max_depth=6, min_child_weight=25, subsample=0.7, colsample_bytree=0.7, n_jobs=8, random_state=s, tree_method='hist')
        elif model_type=='tweedie':
            m = xgb.XGBRegressor(objective='reg:tweedie', tweedie_variance_power=tweedie_p, n_estimators=rounds, learning_rate=lr,
                                 max_depth=6, min_child_weight=25, subsample=0.7, colsample_bytree=0.7, n_jobs=8, random_state=s, tree_method='hist')
        else:
            m = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=rounds, learning_rate=lr,
                                 max_depth=6, min_child_weight=25, subsample=0.7, colsample_bytree=0.7, n_jobs=8, random_state=s, tree_method='hist')
        m.fit(Xtr, yt, sample_weight=wt)
        p = m.predict(te[FEATS])
        if logt: p = np.expm1(p)
        preds.append(np.clip(p,0,None))
    P = np.mean(preds,axis=0)
    mae = np.abs(P-te['future_spend_4w']).values.mean()
    print(f"{model_type} a={alpha} tp={tweedie_p} logt={logt} r={rounds}: honest403/431 MAE={mae:.3f} bias={(P-te['future_spend_4w'].values).mean():.1f} ({time.time()-t0:.0f}s)")
    return P

Pq = run('quantile')
Pl = run('squared', logt=True)
Pt = run('tweedie', tweedie_p=1.3)
Pt2 = run('tweedie', tweedie_p=1.1)
print("blend q+logl:", np.abs(0.5*Pq+0.5*Pl-te['future_spend_4w'].values).mean().round(3))
print("blend q+tweedie1.3:", np.abs(0.5*Pq+0.5*Pt-te['future_spend_4w'].values).mean().round(3))
print("blend q+logl+t1.3+t1.1:", np.abs((Pq+Pl+Pt+Pt2)/4-te['future_spend_4w'].values).mean().round(3))
