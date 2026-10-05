
import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
m = tt.merge(feats, on=['household_key','snapshot_day'])
DROP = ['household_key','snapshot_day','snap_day','snap_week','snap_cycle_pos']
FCOLS = [c for c in feats.columns if c not in DROP]
tr = m[m.snapshot_day<=375]; ev = m[m.snapshot_day>=403]
Xtr, ytr = tr[FCOLS], tr.future_spend_4w
Xev, yev = ev[FCOLS], ev.future_spend_4w
print("train rows", len(tr), "eval rows", len(ev), "eval MAE of median const:", np.abs(np.median(ytr)-yev).mean())

def fit(params, Xtr, ytr, num=400):
    mdl = xgb.XGBRegressor(n_estimators=num, tree_method='hist', n_jobs=4, verbosity=0, **params)
    mdl.fit(Xtr, ytr)
    return mdl

base = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, objective='reg:quantileerror', quantile_alpha=0.5)
mdl = fit(base, Xtr, ytr)
p = mdl.predict(Xev)
print("A ref quantile: MAE", np.abs(p-yev).mean())
for cap in [400,500,600,800]:
    print(f"  clip@{cap}:", round(np.abs(np.clip(p,None,cap)-yev).mean(),3))

# C: log1p target quantile
mdlC = fit(base, Xtr, np.log1p(ytr))
pC = np.expm1(mdlC.predict(Xev))
print("C log1p quantile: MAE", np.abs(pC-yev).mean())
for cap in [400,600]:
    print(f"  clip@{cap}:", round(np.abs(np.clip(pC,None,cap)-yev).mean(),3))

# D: two-part: classifier gate * median pred
from xgboost import XGBClassifier
clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, min_child_weight=40, subsample=0.9, colsample_bytree=0.8, tree_method='hist', n_jobs=4, verbosity=0)
clf.fit(Xtr, (ytr>0).astype(int))
gate = clf.predict_proba(Xev)[:,1]
print("D two-part gate*pred: MAE", np.abs(gate*p-yev).mean(), "| gate*clip600:", np.abs(gate*np.clip(p,None,600)-yev).mean())

# E: blend with exp4w_blend
eb = ev.set_index(['household_key'])['exp4w_blend']
pe = pd.Series(p, index=ev.household_key)
for w in [0.7,0.8,0.9]:
    print(f"E blend w={w}:", round(np.abs(w*p+(1-w)*eb.values-yev).mean(),3))
