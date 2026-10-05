
import agent_api as api, pandas as pd, numpy as np

print(api.snapshot_days())
tt = api.train_targets()
print(tt.shape)
print(tt.future_spend_4w.describe())
print("zero frac:", (tt.future_spend_4w==0).mean())

snap = api.snapshot()
tr = snap.transactions
print(tr.shape, tr.columns.tolist())
print(tr.head())
print("n households:", tr.household_key.nunique(), "day range:", tr.day.min(), tr.day.max())

bf = api.baseline_features()
print(bf.shape, bf.columns.tolist())
print(bf.head(3))


# ---- cell ----

import agent_api as api, pandas as pd, numpy as np

tt = api.train_targets()
snap = api.snapshot(459)
tr = snap.transactions

# lagged spend features per household
def lag_spend(as_of, w):
    d = tr[(tr.day > as_of - w) & (tr.day <= as_of)]
    return d.groupby('household_key').sales_value.sum()

hh = tt.household_key.unique()
res = {}
for as_of in [95,123,151,179,207,235,263,291,319,347,375,403,431]:
    sub = tt[tt.snapshot_day==as_of].set_index('household_key')
    for w in [28,56,112]:
        s = lag_spend(as_of, w).reindex(sub.index).fillna(0)
        res.setdefault(f'lag{w}', []).extend(s.values)
    res.setdefault('y', []).extend(sub.future_spend_4w.values)

for k in res:
    if k!='y':
        print(k, np.corrcoef(res[k], res['y'])[0,1], np.mean(np.abs(np.array(res[k])-np.array(res['y']))))


# ---- cell ----

import agent_api as api, pandas as pd, numpy as np

def make_feats(view, snapshot_day):
    tr = view.table("transactions")
    hh = pd.Index(view.households, name="household_key")
    out = pd.DataFrame(index=hh)
    d = tr[tr.day <= snapshot_day]
    g = d.groupby('household_key')

    for w in [28, 56, 84, 112, 224]:
        s = d[d.day > snapshot_day - w].groupby('household_key').sales_value.sum()
        out[f'spend_{w}'] = s.reindex(hh).fillna(0.0)
        b = d[d.day > snapshot_day - w].groupby('household_key').basket_id.nunique()
        out[f'baskets_{w}'] = b.reindex(hh).fillna(0.0)

    # trend ratios
    out['trend_28_56'] = out['spend_28'] / (out['spend_56']/2 + 1)
    out['trend_56_112'] = out['spend_56'] / (out['spend_112']/2 + 1)
    # per-basket value
    out['abv_28'] = out['spend_28'] / (out['baskets_28'] + 1)
    # recency & tenure
    out['days_since_last'] = snapshot_day - g.day.max().reindex(hh)
    out['days_since_first'] = snapshot_day - g.day.min().reindex(hh)
    out['active_28'] = (out['baskets_28'] > 0).astype(float)
    # discount usage last 84d
    d84 = d[d.day > snapshot_day - 84]
    out['coupon_disc_84'] = d84.groupby('household_key').coupon_disc.sum().reindex(hh).fillna(0)
    out['retail_disc_84'] = d84.groupby('household_key').retail_disc.sum().reindex(hh).fillna(0)
    # distinct stores/products last 84d
    out['n_stores_84'] = d84.groupby('household_key').store_id.nunique().reindex(hh).fillna(0)
    out['n_prods_84'] = d84.groupby('household_key').product_id.nunique().reindex(hh).fillna(0)
    # demographics
    demo = view.table("demographics")
    if demo is not None and len(demo):
        demo = demo.set_index('household_key')
        for c in demo.columns:
            out['dem_'+c] = demo[c].reindex(hh).astype('category').cat.codes.replace(-1, np.nan)
    out['has_demo'] = out.index.isin(demo.index).astype(float) if demo is not None and len(demo) else 0.0
    # calendar
    out['snapshot_day'] = float(snapshot_day)
    out['week_of_year'] = ((snapshot_day + 8) // 7) % 52
    return out

feats = api.build_features(make_feats)
print(feats.shape, feats.columns.tolist())
print(feats.head(3))
api.save_table(feats, "feats_v1")


# ---- cell ----

import agent_api as api, pandas as pd, numpy as np, xgboost as xgb

feats = api.load_saved("feats_v1")
tt = api.train_targets()
tr = feats[feats.snapshot_day.isin(api.snapshot_days()['train'])].merge(tt, on=['household_key','snapshot_day'])
va = feats[feats.snapshot_day.isin(api.snapshot_days()['validation'])]
drop = ['household_key','snapshot_day','future_spend_4w']
Xtr, ytr = tr.drop(columns=drop), tr.future_spend_4w
Xva = va.drop(columns=drop)

model = xgb.XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=10, reg_lambda=5, tree_method='hist',
                         n_jobs=8, random_state=0)
model.fit(Xtr, ytr)
pred = model.predict(Xva)
sub = va[['household_key','snapshot_day']].copy()
sub['prediction'] = pred
print(sub.prediction.describe())
path = api.save_table(sub, "pred_e001")
print(path)


# ---- cell ----

import agent_api as api, pandas as pd, numpy as np, xgboost as xgb

feats = api.load_saved("feats_v1.parquet")
tt = api.train_targets()
tr = feats[feats.snapshot_day.isin(api.snapshot_days()['train'])].merge(tt, on=['household_key','snapshot_day'])
va = feats[feats.snapshot_day.isin(api.snapshot_days()['validation'])]
drop = ['household_key','snapshot_day','future_spend_4w']
Xtr, ytr = tr.drop(columns=drop), tr.future_spend_4w
Xva = va.drop(columns=drop)

model = xgb.XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=10, reg_lambda=5, tree_method='hist',
                         n_jobs=8, random_state=0)
model.fit(Xtr, ytr)
pred = model.predict(Xva)
sub = va[['household_key','snapshot_day']].copy()
sub['prediction'] = pred
print(sub.prediction.describe())
path = api.save_table(sub, "pred_e001")
print(path)


# ---- cell ----

import agent_api as api, pandas as pd, numpy as np, xgboost as xgb

feats = api.load_saved("feats_v1.parquet")
tt = api.train_targets()
tr = feats[feats.snapshot_day.isin(api.snapshot_days()['train'])].merge(tt, on=['household_key','snapshot_day'])
va = feats[feats.snapshot_day.isin(api.snapshot_days()['validation'])]
drop = ['household_key','snapshot_day','future_spend_4w']
Xtr, ytr = tr.drop(columns=drop), tr.future_spend_4w
Xva = va.drop(columns=['household_key','snapshot_day'])

model = xgb.XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=10, reg_lambda=5, tree_method='hist',
                         n_jobs=8, random_state=0)
model.fit(Xtr, ytr)
pred = model.predict(Xva)
sub = va[['household_key','snapshot_day']].copy()
sub['prediction'] = pred
print(sub.prediction.describe())
path = api.save_table(sub, "pred_e001")
print(path)
