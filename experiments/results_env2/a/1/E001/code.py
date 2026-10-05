import agent_api as A
print(A.snapshot_days())
print(A.KEYS, A.TARGET)
s = A.snapshot()
print("day", s.day, "week", s.week)
print("n households", len(s.households))
print(A.describe_tables())


# ---- cell ----
import agent_api as A
s = A.snapshot()
attrs = [a for a in dir(s) if not a.startswith('_')]
print(attrs)
print("day", s.day, "week", s.week)
tr = s.table("transactions")
print(tr.shape)
print(tr['day'].max())


# ---- cell ----
import agent_api as A
tr = A.snapshot().transactions
g = tr.groupby('household_key')['sales_value'].agg(['sum','count','nunique'])
print(g.describe())
# spend per 28d window per household
tr['w'] = (tr['day']-1)//28
w = tr.groupby(['household_key','w'])['sales_value'].sum()
print(w.describe())
print("share of windows with zero spend:", (w==0).mean())
# how many households have a row at a snapshot? rule: first purchase >= 84 days earlier
first = tr.groupby('household_key')['day'].min()
print(first.describe())


# ---- cell ----
import agent_api as A
import numpy as np, pandas as pd, xgboost as xgb

VAL_DAYS = A.snapshot_days()['validation']

def encode_demo(demo, hh):
    demo = demo.set_index('household_key')
    out = pd.DataFrame(index=hh)
    out['has_demo'] = out.index.isin(demo.index).astype(int)
    def ord_map(col, pat):
        if col not in demo: return pd.Series(0, index=hh)
        s = demo[col].reindex(hh).fillna('')
        return s.str.extract(r'(\d+)')[0].astype(float).fillna(0)
    out['age'] = ord_map('classification_1', None)
    out['cls3'] = ord_map('classification_3', None)
    c4 = demo['classification_4'].reindex(hh).fillna('') if 'classification_4' in demo else pd.Series('', index=hh)
    out['hh_size'] = c4.str.extract(r'(\d+)')[0].astype(float).fillna(0)
    out['cls5'] = ord_map('classification_5', None)
    for col, name in [('classification_2','cls2'), ('homeowner_desc','owner'), ('kid_category_desc','kids')]:
        if col in demo:
            out[name] = demo[col].reindex(hh).astype('category').cat.codes.replace(-1, np.nan).fillna(-1)
        else:
            out[name] = -1
    return out

def make_features(view, snapshot_day):
    hh = pd.Index(view.households, name='household_key')
    tr = view.transactions
    d = tr['day']
    f = {}
    for k, nm in [(28,'s28'),(56,'s56'),(84,'s84'),(112,'s112'),(364,'s364')]:
        f[nm] = tr[d > snapshot_day-k].groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
    b = tr.drop_duplicates('basket_id')
    for k, nm in [(28,'b28'),(56,'b56'),(84,'b84')]:
        f[nm] = b[b['day'] > snapshot_day-k].groupby('household_key').size().reindex(hh, fill_value=0)
    f['avg_basket_28'] = (f['s28']/f['b28'].replace(0, np.nan)).fillna(0)
    f['recency'] = snapshot_day - tr.groupby('household_key')['day'].max().reindex(hh)
    f['tenure'] = snapshot_day - tr.groupby('household_key')['day'].min().reindex(hh)
    f['trend'] = f['s28']/(f['s56']-f['s28']).clip(lower=0)+1
    f['total_spend'] = tr.groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
    f['total_baskets'] = b.groupby('household_key').size().reindex(hh, fill_value=0)
    X = pd.DataFrame(f)
    X = X.join(encode_demo(view.demographics, hh))
    X['snap_day'] = snapshot_day
    X['week'] = (snapshot_day+8)//7
    X['week_sin'] = np.sin(2*np.pi*X['week']/52); X['week_cos'] = np.cos(2*np.pi*X['week']/52)
    return X.astype(float)

feats = A.build_features(make_features)
print(feats.shape, feats.columns.tolist())
tt = A.train_targets()
train = feats.merge(tt, on=A.KEYS, how='inner')
val = feats[feats['snapshot_day'].isin(VAL_DAYS)]
print("train", train.shape, "val", val.shape)
fc = [c for c in feats.columns if c not in A.KEYS]
Xtr, ytr = train[fc].values, train['future_spend_4w'].values
m = xgb.XGBRegressor(n_estimators=900, learning_rate=0.04, max_depth=6, subsample=0.8,
                     colsample_bytree=0.8, min_child_weight=5, n_jobs=8, random_state=0)
m.fit(Xtr, ytr)
pred = m.predict(val[fc].values)
out = val[A.KEYS].copy(); out['prediction'] = pred
path = A.save_table(out, 'e001_preds')
print(path)
imp = sorted(zip(fc, m.feature_importances_), key=lambda t:-t[1])
print([ (k, round(v,3)) for k,v in imp[:15] ])


# ---- cell ----
import agent_api as A
import numpy as np, pandas as pd, xgboost as xgb

VAL_DAYS = A.snapshot_days()['validation']

def encode_demo(demo, hh):
    demo = demo.set_index('household_key')
    out = pd.DataFrame(index=hh)
    out['has_demo'] = out.index.isin(demo.index).astype(int)
    for col, name in [('classification_1','age'), ('classification_3','cls3'),
                      ('classification_4','hh_size'), ('classification_5','cls5')]:
        if col in demo.columns:
            s = demo[col].reindex(hh).astype(object).where(demo[col].reindex(hh).notna(), '')
            out[name] = s.str.extract(r'(\d+)')[0].astype(float).fillna(0)
        else:
            out[name] = 0.0
    for col, name in [('classification_2','cls2'), ('homeowner_desc','owner'), ('kid_category_desc','kids')]:
        if col in demo.columns:
            out[name] = demo[col].reindex(hh).astype('category').cat.codes.replace(-1, np.nan).fillna(-1)
        else:
            out[name] = -1.0
    return out

def make_features(view, snapshot_day):
    hh = pd.Index(view.households, name='household_key')
    tr = view.transactions
    d = tr['day']
    f = {}
    for k, nm in [(28,'s28'),(56,'s56'),(84,'s84'),(112,'s112'),(364,'s364')]:
        f[nm] = tr[d > snapshot_day-k].groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
    b = tr.drop_duplicates('basket_id')
    for k, nm in [(28,'b28'),(56,'b56'),(84,'b84')]:
        f[nm] = b[b['day'] > snapshot_day-k].groupby('household_key').size().reindex(hh, fill_value=0)
    f['avg_basket_28'] = (f['s28']/f['b28'].replace(0, np.nan)).fillna(0)
    f['recency'] = snapshot_day - tr.groupby('household_key')['day'].max().reindex(hh)
    f['tenure'] = snapshot_day - tr.groupby('household_key')['day'].min().reindex(hh)
    f['trend'] = f['s28']/(f['s56']-f['s28']).clip(lower=0)+1
    f['total_spend'] = tr.groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
    f['total_baskets'] = b.groupby('household_key').size().reindex(hh, fill_value=0)
    X = pd.DataFrame(f)
    X = X.join(encode_demo(view.demographics, hh))
    X['snap_day'] = snapshot_day
    X['week'] = (snapshot_day+8)//7
    X['week_sin'] = np.sin(2*np.pi*X['week']/52); X['week_cos'] = np.cos(2*np.pi*X['week']/52)
    return X.astype(float)

feats = A.build_features(make_features)
print(feats.shape)
tt = A.train_targets()
train = feats.merge(tt, on=A.KEYS, how='inner')
val = feats[feats['snapshot_day'].isin(VAL_DAYS)]
print("train", train.shape, "val", val.shape)
fc = [c for c in feats.columns if c not in A.KEYS]
Xtr, ytr = train[fc].values, train['future_spend_4w'].values
m = xgb.XGBRegressor(n_estimators=900, learning_rate=0.04, max_depth=6, subsample=0.8,
                     colsample_bytree=0.8, min_child_weight=5, n_jobs=8, random_state=0)
m.fit(Xtr, ytr)
pred = m.predict(val[fc].values)
out = val[A.KEYS].copy(); out['prediction'] = pred
path = A.save_table(out, 'e001_preds')
print(path)
imp = sorted(zip(fc, m.feature_importances_), key=lambda t:-t[1])
print([ (k, round(v,3)) for k,v in imp[:15] ])


# ---- cell ----
import agent_api as A
import numpy as np, pandas as pd, xgboost as xgb

VAL_DAYS = A.snapshot_days()['validation']

def encode_demo(demo, hh):
    demo = demo.set_index('household_key')
    out = pd.DataFrame(index=hh)
    out['has_demo'] = out.index.isin(demo.index).astype(int)
    for col, name in [('classification_1','age'), ('classification_3','cls3'),
                      ('classification_4','hh_size'), ('classification_5','cls5')]:
        if col in demo.columns:
            s = demo[col].reindex(hh).astype(object).where(demo[col].reindex(hh).notna(), '')
            out[name] = s.str.extract(r'(\d+)')[0].astype(float).fillna(0)
        else:
            out[name] = 0.0
    for col, name in [('classification_2','cls2'), ('homeowner_desc','owner'), ('kid_category_desc','kids')]:
        if col in demo.columns:
            out[name] = demo[col].reindex(hh).astype('category').cat.codes.replace(-1, np.nan).fillna(-1)
        else:
            out[name] = -1.0
    return out

def make_features(view, snapshot_day):
    hh = pd.Index(view.households, name='household_key')
    tr = view.transactions
    d = tr['day']
    f = {}
    for k, nm in [(28,'s28'),(56,'s56'),(84,'s84'),(112,'s112'),(364,'s364')]:
        f[nm] = tr[d > snapshot_day-k].groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
    b = tr.drop_duplicates('basket_id')
    for k, nm in [(28,'b28'),(56,'b56'),(84,'b84')]:
        f[nm] = b[b['day'] > snapshot_day-k].groupby('household_key').size().reindex(hh, fill_value=0)
    f['avg_basket_28'] = (f['s28']/f['b28'].replace(0, np.nan)).fillna(0)
    f['recency'] = snapshot_day - tr.groupby('household_key')['day'].max().reindex(hh)
    f['tenure'] = snapshot_day - tr.groupby('household_key')['day'].min().reindex(hh)
    prev = (f['s56']-f['s28'])
    f['trend'] = f['s28']/prev.replace(0, np.nan)
    f['trend'] = f['trend'].replace([np.inf,-np.inf], np.nan).fillna(1.0).clip(0, 5)
    f['total_spend'] = tr.groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
    f['total_baskets'] = b.groupby('household_key').size().reindex(hh, fill_value=0)
    X = pd.DataFrame(f)
    X = X.join(encode_demo(view.demographics, hh))
    X['snap_day'] = snapshot_day
    X['week'] = (snapshot_day+8)//7
    X['week_sin'] = np.sin(2*np.pi*X['week']/52); X['week_cos'] = np.cos(2*np.pi*X['week']/52)
    return X.replace([np.inf,-np.inf], np.nan).fillna(0).astype(float)

feats = A.build_features(make_features)
tt = A.train_targets()
train = feats.merge(tt, on=A.KEYS, how='inner')
val = feats[feats['snapshot_day'].isin(VAL_DAYS)]
print("train", train.shape, "val", val.shape)
fc = [c for c in feats.columns if c not in A.KEYS]
Xtr, ytr = train[fc].values, train['future_spend_4w'].values
m = xgb.XGBRegressor(n_estimators=900, learning_rate=0.04, max_depth=6, subsample=0.8,
                     colsample_bytree=0.8, min_child_weight=5, n_jobs=8, random_state=0)
m.fit(Xtr, ytr)
pred = m.predict(val[fc].values)
out = val[A.KEYS].copy(); out['prediction'] = pred
path = A.save_table(out, 'e001_preds')
print(path)
imp = sorted(zip(fc, m.feature_importances_), key=lambda t:-t[1])
print([ (k, round(v,3)) for k,v in imp[:15] ])
