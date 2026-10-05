
import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
m = tt.merge(feats, on=['household_key','snapshot_day'])
DROP = ['household_key','snapshot_day','snap_day','snap_week','snap_cycle_pos']
for c in m.columns:
    if str(m[c].dtype)=='category': m[c] = m[c].cat.codes
FCOLS = [c for c in feats.columns if c not in DROP]
tr = m[m.snapshot_day<=375]; ev = m[m.snapshot_day>=403]
Xtr, ytr = tr[FCOLS].values.astype(float), tr.future_spend_4w.values
Xev, yev = ev[FCOLS].values.astype(float), ev.future_spend_4w.values
base = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, objective='reg:quantileerror', quantile_alpha=0.5)
mdl = xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, **base).fit(Xtr, ytr)
pev = mdl.predict(Xev)
print("refit encoded MAE:", round(np.abs(pev-yev).mean(),4))

def ev_mae(nc_tr, nc_ev, name):
    Xtr2 = np.column_stack([Xtr, np.asarray(nc_tr, dtype=float).reshape(len(tr),-1)])
    Xev2 = np.column_stack([Xev, np.asarray(nc_ev, dtype=float).reshape(len(ev),-1)])
    mm = xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, **base).fit(Xtr2, ytr)
    print(name, "MAE:", round(np.abs(mm.predict(Xev2)-yev).mean(),4))

lagcols=['lag1_spend','lag2_spend','lag3_spend']
ev_mae(tr[lagcols].std(axis=1).values, ev[lagcols].std(axis=1).values, "lag_std")
ev_mae(mdl.predict(Xtr)[:,None], pev[:,None], "stack_pred_as_feat")
for a,b in [('spend_28','tenure'),('spend_28','trips_28'),('exp4w_blend','tenure')]:
    ev_mae((tr[a]*tr[b]).values, (ev[a]*ev[b]).values, f"int_{a}x{b}")
ev_mae((tr.spend_all/np.maximum(tr.tenure,1)).values, (ev.spend_all/np.maximum(ev.tenure,1)).values, "spend_per_day_all")
