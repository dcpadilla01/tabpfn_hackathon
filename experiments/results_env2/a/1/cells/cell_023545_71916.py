import pandas as pd, numpy as np
pd.set_option('display.width', 250)
allF = agent_api.load_saved('allF.parquet')
tt = agent_api.train_targets()
# NOTE: train_targets only covers train snapshots
print('train_targets snapshots:', sorted(tt.snapshot_day.unique()))
tr = allF[allF.snapshot_day<=431].copy()
y = tr['future_spend_4w']
from sklearn.metrics import mean_absolute_error as mae

# internal holdout: fit on snapshots <=375, evaluate on 403+431
fit = tr[tr.snapshot_day<=375]
ho  = tr[tr.snapshot_day>=403]
print('fit rows', len(fit), 'holdout rows', len(ho))

def rep(name, yt, yp):
    print('%-28s MAE %.3f  bias %+.2f' % (name, np.abs(yt-yp).mean(), (yp-yt).mean()))

# 1) simple baselines on holdout
rep('spend_28', ho.future_spend_4w, ho.spend_28)
rep('spend_84/3', ho.future_spend_4w, ho.spend_84/3)
rep('spend_56/2', ho.future_spend_4w, ho.spend_56/2)
rep('blend .5', ho.future_spend_4w, .5*ho.spend_28+.5*ho.spend_84/3)
rep('blend .35', ho.future_spend_4w, .35*ho.spend_28+.65*ho.spend_84/3)
rep('spend_112/4', ho.future_spend_4w, ho.spend_112/4)
rep('spend_168/6', ho.future_spend_4w, ho.spend_168/6)

# 2) linear calibration of spend_28 fitted on fit-set
b = np.polyfit(fit.spend_28, fit.future_spend_4w, 1)
print('linear fit slope/intercept', b.round(4))
rep('lin(spend_28)', ho.future_spend_4w, np.clip(np.polyval(b, ho.spend_28), 0, None))
# isotonic
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds='clip').fit(fit.spend_28, fit.future_spend_4w)
rep('iso(spend_28)', ho.future_spend_4w, iso.predict(ho.spend_28))
# quantile-ish: median regression via isotonic on quantile
iso5 = IsotonicRegression(out_of_bounds='clip').fit(fit.spend_28, fit.future_spend_4w)
# 3) two-feature OLS: spend_28, spend_84/3
X = np.column_stack([fit.spend_28, fit.spend_84/3, np.ones(len(fit))])
coef, *_ = np.linalg.lstsq(X, fit.future_spend_4w, rcond=None)
print('ols coef', coef.round(4))
Xh = np.column_stack([ho.spend_28, ho.spend_84/3, np.ones(len(ho))])
rep('ols2', ho.future_spend_4w, np.clip(Xh@coef,0,None))
# 4) gradient boosting light on 2 feats
import xgboost as xgb
m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, objective='reg:absoluteerror', n_jobs=8)
m.fit(fit[['spend_28','spend_84','spend_364','actdays_28','actdays_84','recency','tenure']].fillna(0), fit.future_spend_4w)
rep('xgb-abs small', ho.future_spend_4w, m.predict(ho[['spend_28','spend_84','spend_364','actdays_28','actdays_84','recency','tenure']].fillna(0)))