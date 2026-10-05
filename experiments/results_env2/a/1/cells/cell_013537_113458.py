
import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings('ignore')

F = agent_api.load_saved('allF.parquet')
TARGET = 'future_spend_4w'
DROP = [TARGET]
FEATS = [c for c in F.columns if c not in DROP]
TRAIN_DAYS = [95,123,151,179,207,235,263,291,319,347,375,403,431]

def decay_weights(days, half=140):
    age = (431 - days).astype(float)
    return 0.5 ** (age / half)

def fit_xgb(Xtr, ytr, wtr, params, num_rounds):
    dtr = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
    bst = xgb.train(params, dtr, num_boost_round=num_rounds, verbose_eval=False)
    return bst

def run_pair(feats_df, params, half=140, rounds=2400, lr=0.03, days_train_a=None, days_train_b=None, verbose=True):
    feats_df = feats_df.copy()
    y = feats_df[TARGET].values
    X = feats_df[FEATS].astype(float)
    days = feats_df['snapshot_day']
    dA = days_train_a or [d for d in TRAIN_DAYS if d <= 403]
    dB = days_train_b or [d for d in TRAIN_DAYS if d <= 375]
    pa = dict(params); pa['learning_rate'] = lr
    t0=time.time()
    mA = fit_xgb(X[days.isin(dA)], y[days.isin(dA)], decay_weights(days[days.isin(dA)], half), pa, rounds)
    mB = fit_xgb(X[days.isin(dB)], y[days.isin(dB)], decay_weights(days[days.isin(dB)], half), pa, rounds)
    out = {}
    # eval A on 431, B on 403
    for mdl, ev in [(mA,431),(mB,403)]:
        m = ev==431
        idx = days==ev
        p = mdl.predict(xgb.DMatrix(X[idx]))
        t = y[idx]
        out[ev] = np.abs(p-t).mean()
    if verbose:
        print('  eval MAE: d431=%.3f d403=%.3f proxy=%.3f (%.0fs)' % (out[431], out[403], 0.5*(out[431]+out[403]), time.time()-t0))
    return out, (mA,mB)

base_params = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=6,
                   min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
                   tree_method='hist', eval_metric=['mae'])
res, mdls = run_pair(F, base_params)
