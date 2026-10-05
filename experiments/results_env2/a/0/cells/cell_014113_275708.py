
import pandas as pd, numpy as np, agent_api as A, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')
from sklearn.isotonic import IsotonicRegression

f4 = load_saved('feats_v4.parquet').copy()
f3 = load_saved('feats_v3.parquet').copy()
print('f3 cols not in f4:', [c for c in f3.columns if c not in f4.columns])
tt = train_targets(); train_days = A.snapshot_days()['train']
def prep(df):
    df = df.copy()
    for c in df.columns:
        if c not in ('household_key','snapshot_day') and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.factorize(df[c])[0].astype('float32')
    return df
f4, f3 = prep(f4), prep(f3)
tr4 = f4[f4.snapshot_day.isin(train_days)].merge(tt, on=['household_key','snapshot_day'])
tr3 = f3[f3.snapshot_day.isin(train_days)].merge(tt, on=['household_key','snapshot_day'])
BASE = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, n_estimators=400,
            subsample=0.9, colsample_bytree=0.9, tree_method='hist', n_jobs=8)
def fit_q(Xtr, ytr, u=0.5, sw=None):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=u, **BASE)
    m.fit(Xtr, ytr, sample_weight=sw); return m

def evalset(df, cols, fit_max, eval_days, recency_tau=None, iso_from=None):
    fi = df.snapshot_day <= fit_max; ei = df.snapshot_day.isin(eval_days)
    Xf, yf = df.loc[fi, cols].values.astype(np.float32), df.future_spend_4w.values[fi]
    Xe, ye = df.loc[ei, cols].values.astype(np.float32), df.future_spend_4w.values[ei]
    sd = df.snapshot_day.values[fi]
    sw = np.exp((sd - fit_max)/recency_tau) if recency_tau else None
    m = fit_q(Xf, yf, 0.5, sw); p = m.predict(Xe)
    if iso_from is not None:
        # calibrate using model trained on iso_from (earlier), predicting on eval days
        fj = df.snapshot_day <= iso_from
        mj = fit_q(df.loc[fj, cols].values.astype(np.float32), df.future_spend_4w.values[fj], 0.5)
        pj = mj.predict(Xe)
        iso = IsotonicRegression(out_of_bounds='clip').fit(pj, ye)
        p = iso.predict(p)
    return p, ye

cols4 = [c for c in tr4.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
cols3 = [c for c in tr3.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
print('n cols: f3 %d f4 %d' % (len(cols3), len(cols4)))
res = {}
for tag, df, cols in [('f3', tr3, cols3), ('f4', tr4, cols4)]:
    for tau in [None, 250, 150]:
        p, ye = evalset(df, cols, 403, [431], tau)
        p2, ye2 = evalset(df, cols, 375, [403,431], tau)
        res[(tag,tau)] = (np.abs(p-ye).mean(), np.abs(p2-ye2).mean())
for k,v in res.items():
    print('%-10s tau=%-4s f1 %.2f f2 %.2f avg %.2f' % (k[0], k[1], v[0], v[1], (v[0]+v[1])/2))
# isotonic variant on f4
p, ye = evalset(tr4, cols4, 403, [431], None, iso_from=375)
p2, ye2 = evalset(tr4, cols4, 375, [403,431], None, iso_from=347)
print('f4+iso      f1 %.2f f2 %.2f avg %.2f' % (np.abs(p-ye).mean(), np.abs(p2-ye2).mean(), (np.abs(p-ye).mean()+np.abs(p2-ye2).mean())/2))
