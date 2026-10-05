
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
base = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, objective='reg:quantileerror', quantile_alpha=0.5)
mdl = xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, **base).fit(Xtr, ytr)
pev = mdl.predict(Xev)

def ev_mae(newcols_tr, newcols_ev, name):
    Xtr2 = np.column_stack([Xtr, newcols_tr]); Xev2 = np.column_stack([Xev, newcols_ev])
    mm = xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, **base).fit(Xtr2, ytr)
    print(name, "MAE:", round(np.abs(mm.predict(Xev2)-yev).mean(),4))

# 1. dispersion features: std of the 4 lagged 4w spends
lagcols = ['lag1_spend','lag2_spend','lag3_spend']
for nm, cols in [('lag_std', lagcols), ('lag_std+cv', lagcols+['wk_std8'])]:
    v_tr = tr[cols].std(axis=1).values; v_ev = ev[cols].std(axis=1).values
    ev_mae(v_tr, v_ev, nm)

# 2. blend feature: model prediction as feature (stacking) - use in-sample pred
ev_mae(mdl.predict(Xtr)[:,None], pev[:,None], "stack_pred_as_feat")

# 3. interactions: spend28*tenure, spend28*ntrips
for a,b in [('spend_28','tenure'),('spend_28','trips_28'),('exp4w_blend','tenure')]:
    ev_mae((tr[a]*tr[b]).values[:,None], (ev[a]*ev[b]).values[:,None], f"int_{a}x{b}")

# 4. household-level long-run aggregates: mean/max/std of all past 4w-window spends
# approximate via exp4w_all + spend_all/tenure etc - test adding spend_all/tenure ratio
ev_mae((tr.spend_all/np.maximum(tr.tenure,1)).values[:,None], (ev.spend_all/np.maximum(ev.tenure,1)).values[:,None], "spend_per_day_all")
