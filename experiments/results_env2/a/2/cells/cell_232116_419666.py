
import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')

feats = agent_api.load_saved('feats_v3.parquet')
t = agent_api.train_targets()
tr = feats.merge(t, on=['household_key','snapshot_day'], how='inner')
FCOLS = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
trfit = tr[tr.snapshot_day <= 403]; hold = tr[tr.snapshot_day == 431]
Xtr, ytr = trfit[FCOLS].values.astype(float), trfit['future_spend_4w'].values
Xh, yh = hold[FCOLS].values.astype(float), hold['future_spend_4w'].values

def fit(Xa, ya, seed=1, obj='reg:squarederror', q=None):
    kw = dict(n_estimators=2000, learning_rate=0.02, max_depth=7, min_child_weight=10,
              subsample=0.8, colsample_bytree=0.8, objective=obj, random_state=seed,
              n_jobs=8, tree_method='hist')
    if q is not None: kw['quantile_alpha'] = q
    m = xgb.XGBRegressor(**kw); m.fit(Xa, ya); return m

def ev(Xa, ya, Xb, yb, label):
    m1 = fit(Xa, ya, seed=1); m2 = fit(Xa, ya, seed=2, obj='reg:quantileerror', q=0.5)
    p = 0.5*m1.predict(Xb)+0.5*m2.predict(Xb)
    print(label, "MAE:", round(np.mean(np.abs(p-yb)),3))
    return p

p_base = ev(Xtr, ytr, Xh, yh, "base(92f)")

# ---- new candidate features at snapshot 431 ----
v = agent_api.snapshot(431); tx = v.table('transactions')
hh = hold['household_key'].values
hh_idx = pd.Index(hh)
def wsum(lo, hi):
    d = tx[(tx.day>lo)&(tx.day<=hi)]
    return d.groupby('household_key')['sales_value'].sum().reindex(hh_idx).fillna(0.0).values
s336, s448 = wsum(95,431), wsum(0,431)
s364, s392 = wsum(39,67), wsum(11,39)
# weeks active in last 84d
d84 = tx[(tx.day>347)&(tx.day<=431)]
wk = d84.assign(wk=(d84.day+8)//7).groupby('household_key')['wk'].nunique().reindex(hh_idx).fillna(0).values
# max single-week spend last 84d
ws = d84.assign(wk=(d84.day+8)//7).groupby(['household_key','wk'])['sales_value'].sum().groupby('household_key').max().reindex(hh_idx).fillna(0).values
# weekly spend volatility last 112d
d112 = tx[(tx.day>319)&(tx.day<=431)]
wv = d112.assign(wk=(d112.day+8)//7).groupby(['household_key','wk'])['sales_value'].sum().groupby('household_key').std().reindex(hh_idx).fillna(0).values
# exp decay half-life 56
w = 0.5**((431-tx.day)/56.0)
ed = (tx.sales_value*w).groupby(tx.household_key).sum().reindex(hh_idx).fillna(0).values

NEW = np.column_stack([s336, s448, s364, s392, wk, ws, wv, ed])
Xh2 = np.column_stack([Xh, NEW])
print("new feats corr with y:", [round(np.corrcoef(NEW[:,i], yh)[0,1],3) for i in range(NEW.shape[1])])
ev(np.column_stack([Xtr, np.tile(0,(len(Xtr),8))]), ytr, Xh2, yh, "base+8new(train-na)")  # will fail; skip proper
