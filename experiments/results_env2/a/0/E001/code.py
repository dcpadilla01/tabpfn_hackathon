import numpy as np, pandas as pd
print(api.snapshot_days())
tt = api.train_targets()
print('targets', tt.shape, tt.columns.tolist())
print(tt.future_spend_4w.describe())
print('zero frac', (tt.future_spend_4w==0).mean())
bf = api.baseline_features()
print('baseline', bf.shape, bf.columns.tolist())
v = api.snapshot()
print('day', v.day, 'week', v.week)
tr = v.transactions
print('txn', tr.shape, 'hh', tr.household_key.nunique())
print(tr.head(3).to_string())
hh = v.households
print('households type', type(hh), len(hh))
h0 = list(hh)[0]
h = api.history(h0)
print('hist', h.shape)
print(h.head(3).to_string())
d = v.table('demographics')
print('demo', d.shape)
print(api.describe_tables())


# ---- cell ----
import numpy as np, pandas as pd
print(agent_api.snapshot_days())
tt = agent_api.train_targets()
print('targets', tt.shape, tt.columns.tolist())
print(tt.future_spend_4w.describe())
print('zero frac', (tt.future_spend_4w==0).mean())
bf = agent_api.baseline_features()
print('baseline', bf.shape, bf.columns.tolist())
v = agent_api.snapshot()
print('day', v.day, 'week', v.week)
tr = v.transactions
print('txn', tr.shape, 'hh', tr.household_key.nunique())
print(tr.head(3).to_string())
hh = v.households
print('households type', type(hh), len(hh))
h0 = list(hh)[0]
h = agent_api.history(h0)
print('hist', h.shape)
print(h.head(3).to_string())
d = v.table('demographics')
print('demo', d.shape)
print(agent_api.describe_tables())


# ---- cell ----
import numpy as np, pandas as pd
v = agent_api.snapshot()
hh = v.households
print('households', hh if hh is None else type(hh))
# check via build_features what view.households is
def fn(view, day):
    print('day', day, 'week', view.week, 'n households', len(view.households))
    print(type(view.households))
    raise SystemExit
try:
    agent_api.build_features(fn)
except SystemExit:
    pass
except Exception as e:
    print('ERR', type(e).__name__, e)


# ---- cell ----
import numpy as np, pandas as pd, time

def feats(view, day):
    t = view.table('transactions')
    idx = view.households
    t = t[['household_key','basket_id','day','sales_value','quantity','product_id','store_id',
           'coupon_disc','retail_disc','coupon_match_disc','week_no']]
    out = pd.DataFrame(index=idx)
    g = t.groupby('household_key')
    out['last_day'] = g['day'].max()
    out['first_day'] = g['day'].min()
    def wsum(lo, hi=None):
        m = t['day'] > day - lo
        if hi is not None: m &= t['day'] <= day - hi
        return t[m].groupby('household_key')['sales_value'].sum()
    out['spend_7']   = wsum(7)
    out['spend_28']  = wsum(28)
    out['spend_56']  = wsum(56)
    out['spend_84']  = wsum(84)
    out['spend_168'] = wsum(168)
    out['spend_all'] = g['sales_value'].sum()
    out['spend_prev28'] = wsum(56, 28)
    out['spend_prev56'] = wsum(112, 56)
    def wtrips(lo, hi=None):
        m = t['day'] > day - lo
        if hi is not None: m &= t['day'] <= day - hi
        return t[m].groupby('household_key')['basket_id'].nunique()
    out['trips_28'] = wtrips(28)
    out['trips_84'] = wtrips(84)
    out['trips_prev28'] = wtrips(56, 28)
    # basket-level stats last 84d
    b = t[t['day'] > day-84].groupby(['household_key','basket_id']).agg(
        d=('day','first'), s=('sales_value','sum'), n=('product_id','count'))
    bg = b.groupby(level=0)
    out['basket_mean_84'] = bg['s'].mean()
    out['basket_std_84'] = bg['s'].std()
    out['basket_max_84'] = bg['s'].max()
    out['items_per_basket_84'] = bg['n'].mean()
    out['median_gap_84'] = bg['d'].apply(lambda x: np.median(np.diff(np.sort(x.values))) if len(x) > 1 else np.nan)
    out['active_weeks_84'] = b.reset_index().groupby('household_key')['d'].apply(
        lambda x: x.apply(lambda d: (day - d)//7).nunique())
    out['n_products_84'] = t[t['day'] > day-84].groupby('household_key')['product_id'].nunique()
    out['n_stores_84'] = t[t['day'] > day-84].groupby('household_key')['store_id'].nunique()
    out['coupon_disc_28'] = t[t['day'] > day-28].groupby('household_key')['coupon_disc'].sum().abs()
    out['retail_disc_84'] = t[t['day'] > day-84].groupby('household_key')['retail_disc'].sum().abs()
    out['qty_28'] = t[t['day'] > day-28].groupby('household_key')['quantity'].sum()
    # derived
    out['recency'] = day - out['last_day']
    out['tenure'] = day - out['first_day']
    out['avg_basket_28'] = out['spend_28'] / out['trips_28'].clip(lower=1)
    out['weekly_rate_84'] = out['spend_84'] / 12.0
    out['ratio_28_prev'] = out['spend_28'] / (out['spend_prev28'] + 5.0)
    out['trend_28_84'] = out['spend_28'] / (out['spend_84']/3.0 + 5.0)
    out['spend_28_log'] = np.log1p(out['spend_28'])
    out['spend_84_log'] = np.log1p(out['spend_84'])
    out = out.drop(columns=['last_day','first_day'])
    return out

t0 = time.time()
F = agent_api.build_features(feats)
print('build', time.time()-t0, F.shape)
print(F.columns.tolist())
print(F.isna().mean().sort_values(ascending=False).head(8))
agent_api.save_table(F, 'feats_v1')


# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb
F = agent_api.load_saved('feats_v1')
tt = agent_api.train_targets()
tr = F[F.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403,431])]
va = F[F.snapshot_day.isin([459,487,515,543])]
m = tr.merge(tt, on=['household_key','snapshot_day'])
print(m.shape)
feat_cols = [c for c in F.columns if c not in ('household_key','snapshot_day')]
X, y = m[feat_cols].values, m.future_spend_4w.values
Xv = va[feat_cols].values
mdl = xgb.XGBRegressor(n_estimators=900, learning_rate=0.05, max_depth=7, subsample=0.8,
                       colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
                       objective='reg:absoluteerror', tree_method='hist', n_jobs=8, random_state=0)
mdl.fit(X, y)
pred = mdl.predict(Xv)
out = va[['household_key','snapshot_day']].copy()
out['prediction'] = np.clip(pred, 0, None)
p = agent_api.save_table(out, 'pred_e001')
imp = pd.Series(mdl.feature_importances_, index=feat_cols).sort_values(ascending=False)
print(imp.head(12))


# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb
F = agent_api.load_saved('feats_v1.parquet')
tt = agent_api.train_targets()
tr = F[F.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403,431])]
va = F[F.snapshot_day.isin([459,487,515,543])]
m = tr.merge(tt, on=['household_key','snapshot_day'])
print(m.shape)
feat_cols = [c for c in F.columns if c not in ('household_key','snapshot_day')]
X, y = m[feat_cols].values, m.future_spend_4w.values
Xv = va[feat_cols].values
mdl = xgb.XGBRegressor(n_estimators=900, learning_rate=0.05, max_depth=7, subsample=0.8,
                       colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
                       objective='reg:absoluteerror', tree_method='hist', n_jobs=8, random_state=0)
mdl.fit(X, y)
pred = mdl.predict(Xv)
out = va[['household_key','snapshot_day']].copy()
out['prediction'] = np.clip(pred, 0, None)
p = agent_api.save_table(out, 'pred_e001')
imp = pd.Series(mdl.feature_importances_, index=feat_cols).sort_values(ascending=False)
print(imp.head(12))
