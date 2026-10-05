
import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
m = tt.merge(feats, on=['household_key','snapshot_day'])
FCOLS = [c for c in feats.columns if c not in ['household_key','snapshot_day']]
def enc(df):
    df = df.copy()
    for c in df.columns:
        if str(df[c].dtype)=='category': df[c] = df[c].cat.codes
    return df
Xtr = enc(m[FCOLS]).values.astype(float); ytr = m.future_spend_4w.values
val = feats[feats.snapshot_day>=459].sort_values(['household_key','snapshot_day'])
Xva = enc(val[FCOLS]).values.astype(float)

xgb_params = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, subsample=0.9,
                  colsample_bytree=0.8, reg_lambda=1.0, objective='reg:quantileerror', quantile_alpha=0.55)
mx = xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, **xgb_params).fit(Xtr, ytr)
px = mx.predict(Xva)
mh = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, min_samples_leaf=60,
                                   loss='absolute_error', random_state=0).fit(Xtr, ytr)
ph = mh.predict(Xva)
pred = 0.7*px + 0.3*ph
out = pd.DataFrame({'household_key': val.household_key.values,
                    'snapshot_day': val.snapshot_day.values,
                    'prediction': np.clip(pred, 0, None)})
print(out.shape, out.prediction.describe())
p = agent_api.save_table(out, 'pred_e012.parquet')
print(p)
