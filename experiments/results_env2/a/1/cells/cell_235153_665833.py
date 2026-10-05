import numpy as np, pandas as pd, xgboost as xgb, time
F = agent_api.load_saved('allF.parquet')
FEATS = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
p5 = agent_api.load_saved('e005_preds.parquet')
# E005 per-day MAE on validation
tt = agent_api.train_targets()
m = p5.merge(tt, on=['household_key','snapshot_day'], how='left')
for d,g in m.groupby('snapshot_day'):
    print(d, 'MAE %.3f' % np.abs(g.prediction-g.future_spend_4w).mean(), 'n=%d'%len(g),
          'pred_mean %.1f true_mean %.1f' % (g.prediction.mean(), g.future_spend_4w.mean()))

def decay_w(days, half=140):
    return 0.5 ** ((days.max() - days) / half)

def run_xgb(tr, va, feats, lr=0.03, rounds=2400, half=140, alpha=0.5, depth=0, leaves=31,
            subs=0.8, cols=0.8, mincw=20, seed=7, early=None, verbose=False):
    Xtr, ytr = tr[feats].values, tr.future_spend_4w.values
    w = decay_w(tr.snapshot_day.astype(float).values, half)
    model = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha,
        n_estimators=rounds, learning_rate=lr, max_depth=depth, max_leaves=leaves,
        grow_policy='lossguide', subsample=subs, colsample_bytree=cols,
        min_child_weight=mincw, reg_lambda=1.0, n_jobs=8, random_state=seed, tree_method='hist')
    model.fit(Xtr, ytr, sample_weight=w, eval_set=[(va[feats].values, va.future_spend_4w.values)],
              verbose=False)
    return model, model.predict(va[feats].values)

t0=time.time()
tr = F[(F.snapshot_day<=403) & F.future_spend_4w.notna()]
va = F[F.snapshot_day==431]
mod, pred = run_xgb(tr, va, FEATS)
print('internal 431 MAE: %.3f  (%.0fs)' % (np.abs(pred-va.future_spend_4w.values).mean(), time.time()-t0))
# baseline: predict train median
print('median pred MAE: %.3f' % np.abs(np.full(len(va), tr.future_spend_4w.median())-va.future_spend_4w.values).mean())
