
import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings('ignore')
F = agent_api.load_saved('allF.parquet')
TARGET='future_spend_4w'; FEATS=[c for c in F.columns if c!=TARGET]
TRAIN_DAYS=[95,123,151,179,207,235,263,291,319,347,375,403,431]
base_params = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=6,
                   min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
                   tree_method='hist', eval_metric=['mae'])

def proxy(train_on_log=False, half=140, rounds=2400, lr=0.03, params=None, feats=None, seed=0):
    P = dict(base_params); P.update(params or {}); P['learning_rate']=lr
    if seed: P['seed']=seed
    df = F if feats is None else feats
    X = df[FEATS if feats is None else [c for c in feats.columns if c not in ('household_key','snapshot_day',TARGET)]].astype(float)
    y_raw = df[TARGET].values; y = np.log1p(y_raw) if train_on_log else y_raw
    days = df['snapshot_day']
    out={}
    for tr_max, ev in [(403,431),(375,403)]:
        m = days<=tr_max
        w = 0.5**((tr_max-days[m]).astype(float)/half)
        bst = xgb.train(P, xgb.DMatrix(X[m],label=y[m],weight=w), rounds, verbose_eval=False)
        idx = days==ev
        p = bst.predict(xgb.DMatrix(X[idx]))
        if train_on_log: p = np.expm1(p)
        out[ev]=np.abs(p-y_raw[idx]).mean()
    print('  d431=%.3f d403=%.3f proxy=%.3f' % (out[431],out[403],0.5*(out[431]+out[403])))
    return out

print('linear target (ref):'); proxy()
print('log1p target:'); proxy(train_on_log=True)
