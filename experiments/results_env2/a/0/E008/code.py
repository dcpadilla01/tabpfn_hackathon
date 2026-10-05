import pandas as pd, numpy as np
f3 = agent_api.load_saved('feats_v3.parquet')
print('feats_v3', f3.shape)
print(f3.columns.tolist())
print(f3.groupby('snapshot_day').size())
tt = agent_api.train_targets()
y = tt.future_spend_4w
print('targets', tt.shape)
print(y.describe())
print('zero frac', (y==0).mean())
print('quantiles', y.quantile([.5,.75,.9,.95,.99,.995]).to_dict())
m = tt.merge(f3, on=['household_key','snapshot_day'], how='left')
print('merged', m.shape)
print('na frac top:', m.isna().mean().sort_values(ascending=False).head(8).to_dict())
num = [c for c in m.columns if c not in ('household_key','future_spend_4w') and pd.api.types.is_numeric_dtype(m[c])]
cor = m[num+['future_spend_4w']].corr()['future_spend_4w'].drop('future_spend_4w')
cor = cor.reindex(cor.abs().sort_values(ascending=False).index)
print('top |corr| with target:')
print(cor.head(25))
for c in ['exp4w_blend','lag0_spend','lag1_spend','lag2_spend']:
    if c in m.columns:
        print(c, 'MAE vs target:', (m[c].clip(lower=0)-m['future_spend_4w']).abs().mean().round(2))


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
m = tt.merge(f3, on=['household_key','snapshot_day'], how='left')
DROP = ['index','household_key','snapshot_day','future_spend_4w']
FEATS = [c for c in m.columns if c not in DROP]
m[FEATS] = m[FEATS].fillna(-1)
for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner','kids']:
    m[c] = m[c].astype('category')
cat = ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner','kids']
for c in cat: m[c] = m[c].cat.codes
# CV: train on snaps <= 403, eval on 431 (and 375 as second)
def cv(train_max, eval_snaps, params, blend_w=0.7, feats=None):
    feats = feats or FEATS
    tr = m[m.snapshot_day <= train_max]; te = m[m.snapshot_day.isin(eval_snaps)]
    p = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=4, min_child_weight=20,
             learning_rate=0.05, n_estimators=800, subsample=0.8, colsample_bytree=0.8,
             tree_method='hist', enable_categorical=True, eval_metric=None, n_jobs=4)
    p.update(params)
    mod = xgb.XGBRegressor(**p)
    mod.fit(tr[feats], tr.future_spend_4w)
    pred = np.clip(mod.predict(te[feats]), 0, None)
    if blend_w < 1:
        pred = blend_w*pred + (1-blend_w)*te.exp4w_blend.values
    res = {}
    for s in eval_snaps:
        t = te[te.snapshot_day==s]
        res[s] = (t.future_spend_4w - t.pred).abs().mean() if False else None
    out = te[['snapshot_day']].copy(); out['pred']=pred
    maes = {s: np.abs(out[out.snapshot_day==s].pred.values - te[te.snapshot_day==s].future_spend_4w.values).mean() for s in eval_snaps}
    return maes, np.mean(list(maes.values()))
base, base_m = cv(403, [375,431])
print('E007-config CV', base, round(base_m,3))
# variant: no blend (pure model)
p1, m1 = cv(403, [375,431], blend_w=1.0)
print('pure model', p1, round(m1,3))
# variant: deeper
p2, m2 = cv(403, [375,431], params=dict(max_depth=6, min_child_weight=10, n_estimators=1200))
print('deeper', p2, round(m2,3))
# variant: huber
p3, m3 = cv(403, [375,431], params=dict(objective='reg:pseudohubererror'))
print('huber', p3, round(m3,3))


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
m = tt.merge(f3, on=['household_key','snapshot_day'], how='left')
DROP = ['index','household_key','snapshot_day','future_spend_4w']
FEATS = [c for c in m.columns if c not in DROP]
m[FEATS] = m[FEATS].fillna(-1)
for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner','kids']:
    m[c] = m[c].astype('category').cat.codes
def cv(train_max, eval_snaps, params=None, blend_w=0.7):
    params = params or {}
    tr = m[m.snapshot_day <= train_max]; te = m[m.snapshot_day.isin(eval_snaps)]
    p = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=4, min_child_weight=20,
             learning_rate=0.05, n_estimators=800, subsample=0.8, colsample_bytree=0.8,
             tree_method='hist', n_jobs=4)
    p.update(params)
    mod = xgb.XGBRegressor(**p)
    mod.fit(tr[FEATS], tr.future_spend_4w)
    pred = np.clip(mod.predict(te[FEATS]), 0, None)
    if blend_w < 1:
        pred = blend_w*pred + (1-blend_w)*te.exp4w_blend.values
    maes = {s: round(np.abs(pred[te.snapshot_day==s] - te[te.snapshot_day==s].future_spend_4w.values).mean(),3) for s in eval_snaps}
    return maes, round(np.mean(list(maes.values())),3)
print('E007-config CV', cv(403, [375,431]))
print('pure model  ', cv(403, [375,431], blend_w=1.0))
print('deeper      ', cv(403, [375,431], dict(max_depth=6, min_child_weight=10, n_estimators=1200)))
print('huber       ', cv(403, [375,431], dict(objective='reg:pseudohubererror')))
print('blend 0.5   ', cv(403, [375,431], blend_w=0.5))
print('blend 0.85  ', cv(403, [375,431], blend_w=0.85))


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
m = tt.merge(f3, on=['household_key','snapshot_day'], how='left')
DROP = ['index','household_key','snapshot_day','future_spend_4w']
FEATS = [c for c in m.columns if c not in DROP]
m[FEATS] = m[FEATS].fillna(-1)
for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner','kids']:
    m[c] = m[c].astype('category').cat.codes
def cv(train_max, eval_snaps, params=None, blend_w=1.0):
    params = params or {}
    tr = m[m.snapshot_day <= train_max]; te = m[m.snapshot_day.isin(eval_snaps)]
    p = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=4, min_child_weight=20,
             learning_rate=0.05, n_estimators=800, subsample=0.8, colsample_bytree=0.8,
             tree_method='hist', n_jobs=4)
    p.update(params)
    mod = xgb.XGBRegressor(**p)
    mod.fit(tr[FEATS], tr.future_spend_4w)
    pred = np.clip(mod.predict(te[FEATS]), 0, None)
    if blend_w < 1: pred = blend_w*pred + (1-blend_w)*te.exp4w_blend.values
    maes = {s: round(float(np.abs(pred[te.snapshot_day==s] - te[te.snapshot_day==s].future_spend_4w.values).mean()),3) for s in eval_snaps}
    return maes, round(np.mean(list(maes.values())),3)
t0=time.time()
print('base d4      ', cv(347,[375,431]))
print('d6 mcw10 n1200', cv(347,[375,431], dict(max_depth=6, min_child_weight=10, n_estimators=1200)))
print('d8 mcw10 n1200', cv(347,[375,431], dict(max_depth=8, min_child_weight=10, n_estimators=1200)))
print('d6 mcw20 n1500', cv(347,[375,431], dict(max_depth=6, min_child_weight=20, n_estimators=1500)))
print('d6 lr0.03 n2000', cv(347,[375,431], dict(max_depth=6, min_child_weight=10, learning_rate=0.03, n_estimators=2000)))
print('d6 sqerr     ', cv(347,[375,431], dict(objective='reg:squarederror', max_depth=6, min_child_weight=10, n_estimators=1200)))
print('elapsed', round(time.time()-t0,1))


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, warnings, time
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
m = tt.merge(f3, on=['household_key','snapshot_day'], how='left')
DROP = ['index','household_key','snapshot_day','future_spend_4w']
FEATS = [c for c in m.columns if c not in DROP]
m[FEATS] = m[FEATS].fillna(-1)
for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner','kids']:
    m[c] = m[c].astype('category').cat.codes
tr = m[m.snapshot_day <= 347]; te = m[m.snapshot_day.isin([375,431])]
def xgbfit(df_tr, y, params=None):
    p = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=4, min_child_weight=20,
             learning_rate=0.05, n_estimators=800, subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=4)
    p.update(params or {})
    mod = xgb.XGBRegressor(**p); mod.fit(df_tr[FEATS], y); return mod
def mae(pred):
    return {s: round(float(np.abs(pred[te.snapshot_day==s]-te[te.snapshot_day==s].future_spend_4w.values).mean()),3) for s in [375,431]}, round(float(np.abs(pred-te.future_spend_4w.values).mean()),3)
# A: pure xgb quantile
pA = np.clip(xgbfit(tr, tr.future_spend_4w).predict(te[FEATS]),0,None)
print('xgb quantile      ', mae(pA))
# B: xgb quantile on log1p target
modB = xgbfit(tr, np.log1p(tr.future_spend_4w))
pB = np.clip(np.expm1(modB.predict(te[FEATS])),0,None)
print('xgb quantile log  ', mae(pB))
# C: sklearn HGB absolute_error
modC = HistGradientBoostingRegressor(loss='absolute_error', max_iter=500, learning_rate=0.05, max_depth=None, min_samples_leaf=40, l2_regularization=1.0)
modC.fit(tr[FEATS], tr.future_spend_4w)
pC = np.clip(modC.predict(te[FEATS]),0,None)
print('sklearn HGB abs   ', mae(pC))
# D: ensemble xgb + HGB
print('ens xgb+HGB 50/50 ', mae(0.5*pA+0.5*pC))
print('ens xgb+HGB 70/30 ', mae(0.7*pA+0.3*pC))
print('ens xgb+HGBlog 50 ', mae(0.5*pA+0.5*pB))
print('ens all 3         ', mae((pA+pB+pC)/3))
# E: shrinkage estimator: household median of past cycle spends (lag1-3) shrunk to global
cyc = te[['lag1_spend','lag2_spend','lag3_spend']].values
n = (~np.isnan(cyc)).sum(1) if np.isnan(cyc).any() else (cyc>0).sum(1)
med = np.nanmedian(np.where(cyc>0, cyc, np.nan), axis=1)
gm = tr.future_spend_4w.median()
pE = np.clip((np.nan_to_num(med)*n + gm*3)/(n+3), 0, None)
print('shrinkage est only', mae(pE))


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
m = tt.merge(f3, on=['household_key','snapshot_day'], how='left')
DROP = ['index','household_key','snapshot_day','future_spend_4w']
FEATS = [c for c in m.columns if c not in DROP]
m[FEATS] = m[FEATS].fillna(-1)
for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner','kids']:
    m[c] = m[c].astype('category').cat.codes
tr = m[m.snapshot_day <= 347]; te = m[m.snapshot_day.isin([375,431])].copy()
p = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=4, min_child_weight=20,
         learning_rate=0.05, n_estimators=800, subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=4)
mod = xgb.XGBRegressor(**p); mod.fit(tr[FEATS], tr.future_spend_4w)
te['pred'] = np.clip(mod.predict(te[FEATS]),0,None)
te['y'] = te.future_spend_4w
print('=== by days_since_last bucket ===')
te['dsl_b'] = pd.cut(te.days_since_last, [-1,7,14,21,28,42,60,10000])
g = te.groupby('dsl_b', observed=True).apply(lambda d: pd.Series({'n':len(d),'y_med':d.y.median(),'pred_med':d.pred.median(),'mae':np.abs(d.y-d.pred).mean(),'zero_frac':(d.y==0).mean()}))
print(g.round(2))
print('=== by exp4w_blend decile ===')
te['eb'] = pd.qcut(te.exp4w_blend, 8, duplicates='drop')
g2 = te.groupby('eb', observed=True).apply(lambda d: pd.Series({'n':len(d),'y_med':d.y.median(),'pred_med':d.pred.median(),'mae':np.abs(d.y-d.pred).mean()}))
print(g2.round(2))
print('=== zero-target rows ===')
z = te[te.y==0]
print('n', len(z), 'pred mean', z.pred.mean().round(2), 'pred med', z.pred.median().round(2), 'sum MAE contrib', np.abs(z.y-z.pred).mean().round(2))
nz = te[te.y>0]
print('nonzero n', len(nz), 'MAE', np.abs(nz.y-nz.pred).mean().round(2))
print('=== if pred=0 for dsl>28 ===')
alt = te.pred.copy(); alt[te.days_since_last>28]=0
print('mae', np.abs(alt-te.y).mean().round(3), 'vs', np.abs(te.pred-te.y).mean().round(3))
print('=== if pred=0 for dsl>42 ===')
alt2 = te.pred.copy(); alt2[te.days_since_last>42]=0
print('mae', np.abs(alt2-te.y).mean().round(3))
print('dsl>28 zero frac:', (te[te.days_since_last>28].y==0).mean().round(3), 'n=', (te.days_since_last>28).sum())


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')
prod = agent_api.snapshot(459).products
print(prod.department.value_counts().head(30))
print(prod.brand.value_counts())


# ---- cell ----
import pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')
prod = agent_api.snapshot(459).products
print(prod.commodity_desc.value_counts().head(40))
print('---curr size sample---')
print(prod.curr_size_of_product.head(10).tolist())
t = agent_api.snapshot(459).transactions
print('trans', t.shape)
m = t.merge(prod[['product_id','commodity_desc','brand','department']], on='product_id', how='left')
print('top commodities by sales:')
print(m.groupby('commodity_desc').sales_value.sum().sort_values(ascending=False).head(20))


# ---- cell ----
import pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')
t = agent_api.snapshot(459).transactions
prod = agent_api.snapshot(459).products
m = t.merge(prod[['product_id','commodity_desc']], on='product_id', how='left')
# repeat-purchase structure: fraction of line items that are a re-buy of same commodity in prior 28d
# quick proxy: per household, share of spend in top-3 commodities, and commodity loyalty (repeat rate)
h = m.groupby('household_key')
# commodity concentration (HHI) of spend over trailing 84d
t84 = m[m.day > 459-84]
sp = t84.groupby(['household_key','commodity_desc']).sales_value.sum().reset_index()
tot = sp.groupby('household_key').sales_value.transform('sum')
sp['share'] = sp.sales_value/tot
hhi = (sp.assign(s2=sp.share**2).groupby('household_key').s2.sum())
print('HHI describe:', hhi.describe().round(3).to_dict())
# repeat rate: for each household, fraction of spend in trailing 84d on commodities bought in previous 84d window (84-168)
t168 = m[(m.day > 459-168) & (m.day <= 459-84)]
prev_comm = set(zip(t168.household_key, t168.commodity_desc))
sp['is_repeat'] = [ (h,c) in prev_comm for h,c in zip(sp.household_key, sp.commodity_desc)]
rep = sp.groupby('household_key').apply(lambda d: np.average(d.is_repeat, weights=d.sales_value))
print('repeat-rate describe:', rep.describe().round(3).to_dict())
# stock-up indicator: basket size spikes
b = t.groupby(['household_key','basket_id']).agg(sales=('sales_value','sum'), day=('day','first'))
b = b.sort_values(['household_key','day'])
b['prev'] = b.groupby('household_key').sales.shift(1)
b['ratio'] = b.sales/b.prev
print('basket sales ratio>2.5 frac:', (b.ratio>2.5).mean().round(3))
print('time to check per-household ops feasible')


# ---- cell ----
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

def feats_v4(view, snapshot_day):
    T = view.table('transactions')
    P = view.table('products')
    hh = view.households
    day = snapshot_day
    out = hh.copy()
    # ---- trailing window spends (same as v3 core) ----
    for w in [7,14,28,56,84,168]:
        out[f'spend_{w}'] = T[T.day > day-w].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['spend_all'] = T.groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    # cycle-aligned 4w lags
    for k in [1,2,3]:
        lo, hi = day-28*k-27, day-28*k
        out[f'lag{k}_spend'] = T[(T.day>=lo)&(T.day<=hi)].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['exp4w_blend'] = 0.5*out.lag1_spend + 0.3*out.lag2_spend + 0.2*out.lag3_spend
    out['trend28'] = out.lag1_spend/(out.lag2_spend+5)
    out['trend84'] = out.spend_28/(out.spend_84/3+5)
    # weekly pattern last 8 weeks
    for k in range(8):
        out[f'wk{k}'] = T[(T.day>day-7*(k+1))&(T.day<=day-7*k)].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['active_weeks8'] = (out[[f'wk{k}' for k in range(8)]]>0).sum(1)
    out['wk_mean8'] = out[[f'wk{k}' for k in range(8)]].mean(1)
    out['wk_std8'] = out[[f'wk{k}' for k in range(8)]].std(1).fillna(0)
    out['wk_cv8'] = out.wk_std8/(out.wk_mean8+1)
    # recency / tenure
    out['tenure'] = day - T.groupby('household_key').day.min().reindex(hh)
    out['days_since_last'] = day - T.groupby('household_key').day.max().reindex(hh)
    # trips / baskets
    b = T.groupby(['household_key','basket_id']).agg(sales=('sales_value','sum'), d=('day','max'))
    b = b.sort_values(['household_key','d'])
    b84 = b[b.d > day-84]
    out['trips_84'] = b84.groupby('household_key').size().reindex(hh).fillna(0)
    out['trips_28'] = b[b.d>day-28].groupby('household_key').size().reindex(hh).fillna(0)
    out['basket_mean_84'] = b84.groupby('household_key').sales.mean().reindex(hh)
    out['basket_max_84'] = b84.groupby('household_key').sales.max().reindex(hh)
    out['basket_std_84'] = b84.groupby('household_key').sales.std().reindex(hh)
    out['spend_per_trip28'] = out.spend_28/(out.trips_28+1e-9)
    # gaps
    b['prev_d'] = b.groupby('household_key').d.shift(1)
    b['gap'] = b.d - b.prev_d
    g84 = b[(b.d>day-84) & b.gap.notna()]
    out['gap_mean_84'] = g84.groupby('household_key').gap.mean().reindex(hh)
    out['gap_max_84'] = g84.groupby('household_key').gap.max().reindex(hh)
    # diversity
    out['nprod_84'] = T[T.day>day-84].groupby('household_key').product_id.nunique().reindex(hh).fillna(0)
    out['nstore_84'] = T[T.day>day-84].groupby('household_key').store_id.nunique().reindex(hh).fillna(0)
    out['n_days_active_84'] = T[T.day>day-84].groupby('household_key').day.nunique().reindex(hh).fillna(0)
    # discounts
    out['coupon_disc_84'] = T[T.day>day-84].groupby('household_key').coupon_disc.sum().reindex(hh).fillna(0)
    out['retail_disc_84'] = T[T.day>day-84].groupby('household_key').retail_disc.sum().reindex(hh).fillna(0)
    out['coupon_match_disc_84'] = T[T.day>day-84].groupby('household_key').coupon_match_disc.sum().reindex(hh).fillna(0)
    out['disc_share_84'] = (out.coupon_disc_84+out.retail_disc_84+out.coupon_match_disc_84)/(out.spend_84+1e-9)
    # time of day
    T84 = T[T.day>day-84].copy()
    T84['morn'] = (T84.trans_time<1200).astype(float)
    out['morn_share_84'] = T84.groupby('household_key').morn.mean().reindex(hh)
    # exp blends all-history
    w_all = 0.97**((day-T.day).clip(lower=0))
    out['exp4w_all'] = (T.sales_value*w_all).groupby(T.household_key).sum().reindex(hh).fillna(0)
    T168 = T[T.day>day-168]
    w168 = 0.97**((day-T168.day).clip(lower=0))
    out['exp4w_168'] = (T168.sales_value*w168).groupby(T168.household_key).sum().reindex(hh).fillna(0)
    out['exp4w_84'] = out.spend_84
    out['weekly_rate_all'] = out.spend_all/(out.tenure+1)
    # ================= NEW v4: product-mix / loyalty / stock-up =================
    M = T84.merge(P[['product_id','commodity_desc','brand']], on='product_id', how='left')
    M['commodity_desc'] = M.commodity_desc.fillna('UNK')
    M['brand'] = M.brand.fillna('UNK')
    sp = M.groupby(['household_key','commodity_desc']).sales_value.sum().reset_index()
    tot = sp.groupby('household_key').sales_value.transform('sum')
    sp['share'] = sp.sales_value/tot
    out['hhi_84'] = sp.assign(s2=sp.share**2).groupby('household_key').s2.sum().reindex(hh)
    out['n_commodities_84'] = sp.groupby('household_key').size().reindex(hh).fillna(0)
    out['top_comm_share_84'] = sp.groupby('household_key').share.max().reindex(hh)
    # repeat rate: commodities bought in (84,168] before snapshot also bought in trailing 84
    T168m = T[(T.day>day-168)&(T.day<=day-84)].merge(P[['product_id','commodity_desc']], on='product_id', how='left')
    prev = set(zip(T168m.household_key, T168m.commodity_desc.fillna('UNK')))
    sp['is_rep'] = [(h,c) in prev for h,c in zip(sp.household_key, sp.commodity_desc)]
    out['repeat_rate_84'] = sp.groupby('household_key').apply(lambda d: np.average(d.is_rep, weights=d.sales_value)).reindex(hh)
    # private-brand share
    out['private_share_84'] = (M.assign(pb=(M.brand=='Private').astype(float))
                                .groupby('household_key').apply(lambda d: np.average(d.pb, weights=d.sales_value)).reindex(hh))
    # grocery dept share
    out['grocery_share_84'] = M.assign(g=(M.commodity_desc=='UNK')*0 + (P.set_index('product_id').department.reindex(M.product_id).values=='GROCERY').astype(float)).groupby('household_key').apply(lambda d: np.average(d.g, weights=d.sales_value)).reindex(hh)
    # stock-up: max basket / median basket; frac of baskets with sales > 2.5x prev
    b84r = b84.reset_index().copy()
    b84r['prev_sales'] = b.groupby('household_key').sales.shift(1).reindex(b84.index)
    out['stockup_max_84'] = (b84r.sales/(b84.groupby('household_key').sales.median().reindex(b84r.household_key).values+1e-9)).groupby(b84r.household_key).max().reindex(hh)
    # store loyalty
    st = M.groupby(['household_key','store_id']).sales_value.sum().reset_index()
    out['store_loyalty_84'] = st.groupby('household_key').sales_value.max()/st.groupby('household_key').sales_value.sum().reindex(hh)
    # weekend share
    T84['is_wkend'] = ((T84.day % 7)==5) | ((T84.day % 7)==6)
    out['wkend_share_84'] = T84.groupby('household_key').apply(lambda d: np.average(d.is_wkend.astype(float), weights=d.sales_value)).reindex(hh)
    # demographics
    D = view.table('demographics')
    if D is not None and len(D):
        D = D.set_index('household_key')
        for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc']:
            out[c] = D[c].reindex(hh)
        out['has_demo'] = out.index.isin(D.index).astype(float)
    out['snap_day'] = day
    out['snap_week'] = (day+8)//7
    out['snap_cycle_pos'] = day % 28
    return out

f4 = agent_api.build_features(feats_v4)
agent_api.save_table(f4, 'feats_v4.parquet')
print('saved', f4.shape)
print(f4.columns.tolist())


# ---- cell ----
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

def feats_v4(view, snapshot_day):
    T = view.table('transactions')
    P = view.table('products')
    hh = list(view.households)
    day = snapshot_day
    out = pd.DataFrame(index=hh)
    for w in [7,14,28,56,84,168]:
        out[f'spend_{w}'] = T[T.day > day-w].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['spend_all'] = T.groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    for k in [1,2,3]:
        lo, hi = day-28*k-27, day-28*k
        out[f'lag{k}_spend'] = T[(T.day>=lo)&(T.day<=hi)].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['exp4w_blend'] = 0.5*out.lag1_spend + 0.3*out.lag2_spend + 0.2*out.lag3_spend
    out['trend28'] = out.lag1_spend/(out.lag2_spend+5)
    out['trend84'] = out.spend_28/(out.spend_84/3+5)
    for k in range(8):
        out[f'wk{k}'] = T[(T.day>day-7*(k+1))&(T.day<=day-7*k)].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['active_weeks8'] = (out[[f'wk{k}' for k in range(8)]]>0).sum(1)
    out['wk_mean8'] = out[[f'wk{k}' for k in range(8)]].mean(1)
    out['wk_std8'] = out[[f'wk{k}' for k in range(8)]].std(1).fillna(0)
    out['wk_cv8'] = out.wk_std8/(out.wk_mean8+1)
    out['tenure'] = day - T.groupby('household_key').day.min().reindex(hh)
    out['days_since_last'] = day - T.groupby('household_key').day.max().reindex(hh)
    b = T.groupby(['household_key','basket_id']).agg(sales=('sales_value','sum'), d=('day','max'))
    b = b.sort_values(['household_key','d'])
    b84 = b[b.d > day-84]
    out['trips_84'] = b84.groupby('household_key').size().reindex(hh).fillna(0)
    out['trips_28'] = b[b.d>day-28].groupby('household_key').size().reindex(hh).fillna(0)
    out['basket_mean_84'] = b84.groupby('household_key').sales.mean().reindex(hh)
    out['basket_max_84'] = b84.groupby('household_key').sales.max().reindex(hh)
    out['basket_std_84'] = b84.groupby('household_key').sales.std().reindex(hh)
    out['spend_per_trip28'] = out.spend_28/(out.trips_28+1e-9)
    b['prev_d'] = b.groupby('household_key').d.shift(1)
    b['gap'] = b.d - b.prev_d
    g84 = b[(b.d>day-84) & b.gap.notna()]
    out['gap_mean_84'] = g84.groupby('household_key').gap.mean().reindex(hh)
    out['gap_max_84'] = g84.groupby('household_key').gap.max().reindex(hh)
    out['nprod_84'] = T[T.day>day-84].groupby('household_key').product_id.nunique().reindex(hh).fillna(0)
    out['nstore_84'] = T[T.day>day-84].groupby('household_key').store_id.nunique().reindex(hh).fillna(0)
    out['n_days_active_84'] = T[T.day>day-84].groupby('household_key').day.nunique().reindex(hh).fillna(0)
    out['coupon_disc_84'] = T[T.day>day-84].groupby('household_key').coupon_disc.sum().reindex(hh).fillna(0)
    out['retail_disc_84'] = T[T.day>day-84].groupby('household_key').retail_disc.sum().reindex(hh).fillna(0)
    out['coupon_match_disc_84'] = T[T.day>day-84].groupby('household_key').coupon_match_disc.sum().reindex(hh).fillna(0)
    out['disc_share_84'] = (out.coupon_disc_84+out.retail_disc_84+out.coupon_match_disc_84)/(out.spend_84+1e-9)
    T84 = T[T.day>day-84].copy()
    T84['morn'] = (T84.trans_time<1200).astype(float)
    out['morn_share_84'] = T84.groupby('household_key').morn.mean().reindex(hh)
    w_all = 0.97**((day-T.day).clip(lower=0))
    out['exp4w_all'] = (T.sales_value*w_all).groupby(T.household_key).sum().reindex(hh).fillna(0)
    T168 = T[T.day>day-168]
    w168 = 0.97**((day-T168.day).clip(lower=0))
    out['exp4w_168'] = (T168.sales_value*w168).groupby(T168.household_key).sum().reindex(hh).fillna(0)
    out['exp4w_84'] = out.spend_84
    out['weekly_rate_all'] = out.spend_all/(out.tenure+1)
    # ==== NEW v4: product-mix / loyalty / stock-up ====
    M = T84.merge(P[['product_id','commodity_desc','brand','department']], on='product_id', how='left')
    M['commodity_desc'] = M.commodity_desc.fillna('UNK'); M['brand'] = M.brand.fillna('UNK')
    M['is_grocery'] = (M.department=='GROCERY').astype(float)
    sp = M.groupby(['household_key','commodity_desc']).sales_value.sum().reset_index()
    tot = sp.groupby('household_key').sales_value.transform('sum')
    sp['share'] = sp.sales_value/tot
    out['hhi_84'] = sp.assign(s2=sp.share**2).groupby('household_key').s2.sum().reindex(hh)
    out['n_commodities_84'] = sp.groupby('household_key').size().reindex(hh).fillna(0)
    out['top_comm_share_84'] = sp.groupby('household_key').share.max().reindex(hh)
    T168m = T[(T.day>day-168)&(T.day<=day-84)].merge(P[['product_id','commodity_desc']], on='product_id', how='left')
    prev = set(zip(T168m.household_key, T168m.commodity_desc.fillna('UNK')))
    sp['is_rep'] = [(h,c) in prev for h,c in zip(sp.household_key, sp.commodity_desc)]
    out['repeat_rate_84'] = sp.groupby('household_key').apply(lambda d: np.average(d.is_rep, weights=d.sales_value), include_groups=False).reindex(hh)
    out['private_share_84'] = M.assign(pb=(M.brand=='Private').astype(float)).groupby('household_key').apply(lambda d: np.average(d.pb, weights=d.sales_value), include_groups=False).reindex(hh)
    out['grocery_share_84'] = M.groupby('household_key').apply(lambda d: np.average(d.is_grocery, weights=d.sales_value), include_groups=False).reindex(hh)
    med_b = b84.groupby('household_key').sales.median()
    r = (b84.sales/(b84.household_key.map(med_b)+1e-9))
    out['stockup_max_84'] = r.groupby(b84.index.get_level_values(0)).max().reindex(hh)
    st = M.groupby(['household_key','store_id']).sales_value.sum().reset_index()
    out['store_loyalty_84'] = (st.groupby('household_key').sales_value.max()/st.groupby('household_key').sales_value.sum()).reindex(hh)
    T84['is_wkend'] = ((T84.day % 7)==5) | ((T84.day % 7)==6)
    out['wkend_share_84'] = T84.groupby('household_key').apply(lambda d: np.average(d.is_wkend.astype(float), weights=d.sales_value), include_groups=False).reindex(hh)
    D = view.table('demographics')
    if D is not None and len(D):
        D = D.set_index('household_key')
        for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc']:
            out[c] = D[c].reindex(hh)
        out['has_demo'] = pd.Series(out.index.isin(D.index).astype(float), index=hh)
    out['snap_day'] = day; out['snap_week'] = (day+8)//7; out['snap_cycle_pos'] = day % 28
    return out

f4 = agent_api.build_features(feats_v4)
agent_api.save_table(f4, 'feats_v4.parquet')
print('saved', f4.shape)
new = [c for c in f4.columns if c not in agent_api.load_saved('feats_v3.parquet').columns]
print('new cols:', new)


# ---- cell ----
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

def feats_v4(view, snapshot_day):
    T = view.table('transactions')
    P = view.table('products')
    hh = list(view.households)
    day = snapshot_day
    out = pd.DataFrame(index=hh)
    for w in [7,14,28,56,84,168]:
        out[f'spend_{w}'] = T[T.day > day-w].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['spend_all'] = T.groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    for k in [1,2,3]:
        lo, hi = day-28*k-27, day-28*k
        out[f'lag{k}_spend'] = T[(T.day>=lo)&(T.day<=hi)].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['exp4w_blend'] = 0.5*out.lag1_spend + 0.3*out.lag2_spend + 0.2*out.lag3_spend
    out['trend28'] = out.lag1_spend/(out.lag2_spend+5)
    out['trend84'] = out.spend_28/(out.spend_84/3+5)
    for k in range(8):
        out[f'wk{k}'] = T[(T.day>day-7*(k+1))&(T.day<=day-7*k)].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['active_weeks8'] = (out[[f'wk{k}' for k in range(8)]]>0).sum(1)
    out['wk_mean8'] = out[[f'wk{k}' for k in range(8)]].mean(1)
    out['wk_std8'] = out[[f'wk{k}' for k in range(8)]].std(1).fillna(0)
    out['wk_cv8'] = out.wk_std8/(out.wk_mean8+1)
    out['tenure'] = day - T.groupby('household_key').day.min().reindex(hh)
    out['days_since_last'] = day - T.groupby('household_key').day.max().reindex(hh)
    b = T.groupby(['household_key','basket_id']).agg(sales=('sales_value','sum'), d=('day','max'))
    b = b.sort_values(['household_key','d'])
    b84 = b[b.d > day-84]
    out['trips_84'] = b84.groupby('household_key').size().reindex(hh).fillna(0)
    out['trips_28'] = b[b.d>day-28].groupby('household_key').size().reindex(hh).fillna(0)
    out['basket_mean_84'] = b84.groupby('household_key').sales.mean().reindex(hh)
    out['basket_max_84'] = b84.groupby('household_key').sales.max().reindex(hh)
    out['basket_std_84'] = b84.groupby('household_key').sales.std().reindex(hh)
    out['spend_per_trip28'] = out.spend_28/(out.trips_28+1e-9)
    b['prev_d'] = b.groupby('household_key').d.shift(1)
    b['gap'] = b.d - b.prev_d
    g84 = b[(b.d>day-84) & b.gap.notna()]
    out['gap_mean_84'] = g84.groupby('household_key').gap.mean().reindex(hh)
    out['gap_max_84'] = g84.groupby('household_key').gap.max().reindex(hh)
    out['nprod_84'] = T[T.day>day-84].groupby('household_key').product_id.nunique().reindex(hh).fillna(0)
    out['nstore_84'] = T[T.day>day-84].groupby('household_key').store_id.nunique().reindex(hh).fillna(0)
    out['n_days_active_84'] = T[T.day>day-84].groupby('household_key').day.nunique().reindex(hh).fillna(0)
    out['coupon_disc_84'] = T[T.day>day-84].groupby('household_key').coupon_disc.sum().reindex(hh).fillna(0)
    out['retail_disc_84'] = T[T.day>day-84].groupby('household_key').retail_disc.sum().reindex(hh).fillna(0)
    out['coupon_match_disc_84'] = T[T.day>day-84].groupby('household_key').coupon_match_disc.sum().reindex(hh).fillna(0)
    out['disc_share_84'] = (out.coupon_disc_84+out.retail_disc_84+out.coupon_match_disc_84)/(out.spend_84+1e-9)
    T84 = T[T.day>day-84].copy()
    T84['morn'] = (T84.trans_time<1200).astype(float)
    out['morn_share_84'] = T84.groupby('household_key').morn.mean().reindex(hh)
    w_all = 0.97**((day-T.day).clip(lower=0))
    out['exp4w_all'] = (T.sales_value*w_all).groupby(T.household_key).sum().reindex(hh).fillna(0)
    T168 = T[T.day>day-168]
    w168 = 0.97**((day-T168.day).clip(lower=0))
    out['exp4w_168'] = (T168.sales_value*w168).groupby(T168.household_key).sum().reindex(hh).fillna(0)
    out['exp4w_84'] = out.spend_84
    out['weekly_rate_all'] = out.spend_all/(out.tenure+1)
    # ==== NEW v4 ====
    M = T84.merge(P[['product_id','commodity_desc','brand','department']].astype({'commodity_desc':object,'brand':object,'department':object}), on='product_id', how='left')
    M['commodity_desc'] = M.commodity_desc.astype(object).fillna('UNK'); M['brand'] = M.brand.astype(object).fillna('UNK')
    M['is_grocery'] = (M.department.astype(object).fillna('')=='GROCERY').astype(float)
    sp = M.groupby(['household_key','commodity_desc']).sales_value.sum().reset_index()
    tot = sp.groupby('household_key').sales_value.transform('sum')
    sp['share'] = sp.sales_value/tot
    out['hhi_84'] = sp.assign(s2=sp.share**2).groupby('household_key').s2.sum().reindex(hh)
    out['n_commodities_84'] = sp.groupby('household_key').size().reindex(hh).fillna(0)
    out['top_comm_share_84'] = sp.groupby('household_key').share.max().reindex(hh)
    T168m = T[(T.day>day-168)&(T.day<=day-84)].merge(P[['product_id','commodity_desc']].astype({'commodity_desc':object}), on='product_id', how='left')
    prev = set(zip(T168m.household_key, T168m.commodity_desc.astype(object).fillna('UNK')))
    sp['is_rep'] = [(h,c) in prev for h,c in zip(sp.household_key, sp.commodity_desc)]
    out['repeat_rate_84'] = sp.groupby('household_key').apply(lambda d: np.average(d.is_rep, weights=d.sales_value), include_groups=False).reindex(hh)
    out['private_share_84'] = M.assign(pb=(M.brand=='Private').astype(float)).groupby('household_key').apply(lambda d: np.average(d.pb, weights=d.sales_value), include_groups=False).reindex(hh)
    out['grocery_share_84'] = M.groupby('household_key').apply(lambda d: np.average(d.is_grocery, weights=d.sales_value), include_groups=False).reindex(hh)
    med_b = b84.groupby('household_key').sales.median()
    r = (b84.sales/(b84.index.get_level_values(0).map(med_b)+1e-9))
    out['stockup_max_84'] = r.groupby(b84.index.get_level_values(0)).max().reindex(hh)
    st = M.groupby(['household_key','store_id']).sales_value.sum().reset_index()
    out['store_loyalty_84'] = (st.groupby('household_key').sales_value.max()/st.groupby('household_key').sales_value.sum()).reindex(hh)
    T84['is_wkend'] = ((T84.day % 7)==5) | ((T84.day % 7)==6)
    out['wkend_share_84'] = T84.groupby('household_key').apply(lambda d: np.average(d.is_wkend.astype(float), weights=d.sales_value), include_groups=False).reindex(hh)
    D = view.table('demographics')
    if D is not None and len(D):
        D = D.set_index('household_key')
        for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc']:
            out[c] = D[c].reindex(hh)
        out['has_demo'] = pd.Series(out.index.isin(D.index).astype(float), index=hh)
    out['snap_day'] = day; out['snap_week'] = (day+8)//7; out['snap_cycle_pos'] = day % 28
    return out

f4 = agent_api.build_features(feats_v4)
agent_api.save_table(f4, 'feats_v4.parquet')
print('saved', f4.shape)
new = [c for c in f4.columns if c not in agent_api.load_saved('feats_v3.parquet').columns]
print('new cols:', new)
print(f4[new].describe().T.round(3))


# ---- cell ----
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

def wmean_by(df, key, val, w):
    ww = w.clip(lower=0)
    num = (val*ww).groupby(df[key]).sum()
    den = ww.groupby(df[key]).sum()+1e-9
    return num/den

def feats_v4(view, snapshot_day):
    T = view.table('transactions')
    P = view.table('products')
    hh = list(view.households)
    day = snapshot_day
    out = pd.DataFrame(index=hh)
    for w in [7,14,28,56,84,168]:
        out[f'spend_{w}'] = T[T.day > day-w].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['spend_all'] = T.groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    for k in [1,2,3]:
        lo, hi = day-28*k-27, day-28*k
        out[f'lag{k}_spend'] = T[(T.day>=lo)&(T.day<=hi)].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['exp4w_blend'] = 0.5*out.lag1_spend + 0.3*out.lag2_spend + 0.2*out.lag3_spend
    out['trend28'] = out.lag1_spend/(out.lag2_spend+5)
    out['trend84'] = out.spend_28/(out.spend_84/3+5)
    for k in range(8):
        out[f'wk{k}'] = T[(T.day>day-7*(k+1))&(T.day<=day-7*k)].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['active_weeks8'] = (out[[f'wk{k}' for k in range(8)]]>0).sum(1)
    out['wk_mean8'] = out[[f'wk{k}' for k in range(8)]].mean(1)
    out['wk_std8'] = out[[f'wk{k}' for k in range(8)]].std(1).fillna(0)
    out['wk_cv8'] = out.wk_std8/(out.wk_mean8+1)
    out['tenure'] = day - T.groupby('household_key').day.min().reindex(hh)
    out['days_since_last'] = day - T.groupby('household_key').day.max().reindex(hh)
    b = T.groupby(['household_key','basket_id']).agg(sales=('sales_value','sum'), d=('day','max'))
    b = b.sort_values(['household_key','d'])
    b84 = b[b.d > day-84]
    out['trips_84'] = b84.groupby('household_key').size().reindex(hh).fillna(0)
    out['trips_28'] = b[b.d>day-28].groupby('household_key').size().reindex(hh).fillna(0)
    out['basket_mean_84'] = b84.groupby('household_key').sales.mean().reindex(hh)
    out['basket_max_84'] = b84.groupby('household_key').sales.max().reindex(hh)
    out['basket_std_84'] = b84.groupby('household_key').sales.std().reindex(hh)
    out['spend_per_trip28'] = out.spend_28/(out.trips_28+1e-9)
    b['prev_d'] = b.groupby('household_key').d.shift(1)
    b['gap'] = b.d - b.prev_d
    g84 = b[(b.d>day-84) & b.gap.notna()]
    out['gap_mean_84'] = g84.groupby('household_key').gap.mean().reindex(hh)
    out['gap_max_84'] = g84.groupby('household_key').gap.max().reindex(hh)
    out['nprod_84'] = T[T.day>day-84].groupby('household_key').product_id.nunique().reindex(hh).fillna(0)
    out['nstore_84'] = T[T.day>day-84].groupby('household_key').store_id.nunique().reindex(hh).fillna(0)
    out['n_days_active_84'] = T[T.day>day-84].groupby('household_key').day.nunique().reindex(hh).fillna(0)
    out['coupon_disc_84'] = T[T.day>day-84].groupby('household_key').coupon_disc.sum().reindex(hh).fillna(0)
    out['retail_disc_84'] = T[T.day>day-84].groupby('household_key').retail_disc.sum().reindex(hh).fillna(0)
    out['coupon_match_disc_84'] = T[T.day>day-84].groupby('household_key').coupon_match_disc.sum().reindex(hh).fillna(0)
    out['disc_share_84'] = (out.coupon_disc_84+out.retail_disc_84+out.coupon_match_disc_84)/(out.spend_84+1e-9)
    T84 = T[T.day>day-84].copy()
    T84['morn'] = (T84.trans_time<1200).astype(float)
    out['morn_share_84'] = T84.groupby('household_key').morn.mean().reindex(hh)
    w_all = 0.97**((day-T.day).clip(lower=0))
    out['exp4w_all'] = (T.sales_value*w_all).groupby(T.household_key).sum().reindex(hh).fillna(0)
    T168 = T[T.day>day-168]
    w168 = 0.97**((day-T168.day).clip(lower=0))
    out['exp4w_168'] = (T168.sales_value*w168).groupby(T168.household_key).sum().reindex(hh).fillna(0)
    out['exp4w_84'] = out.spend_84
    out['weekly_rate_all'] = out.spend_all/(out.tenure+1)
    # ==== NEW v4 ====
    M = T84.merge(P[['product_id','commodity_desc','brand','department']], on='product_id', how='left')
    M['commodity_desc'] = M.commodity_desc.astype(object).where(M.commodity_desc.notna(), 'UNK')
    M['brand'] = M.brand.astype(object).where(M.brand.notna(), 'UNK')
    M['is_grocery'] = (M.department=='GROCERY').astype(float)
    sp = M.groupby(['household_key','commodity_desc']).sales_value.sum().reset_index()
    tot = sp.groupby('household_key').sales_value.transform('sum')
    sp['share'] = sp.sales_value/tot
    out['hhi_84'] = sp.assign(s2=sp.share**2).groupby('household_key').s2.sum().reindex(hh)
    out['n_commodities_84'] = sp.groupby('household_key').size().reindex(hh).fillna(0)
    out['top_comm_share_84'] = sp.groupby('household_key').share.max().reindex(hh)
    T168m = T[(T.day>day-168)&(T.day<=day-84)].merge(P[['product_id','commodity_desc']], on='product_id', how='left')
    T168m['commodity_desc'] = T168m.commodity_desc.astype(object).where(T168m.commodity_desc.notna(), 'UNK')
    prev = set(zip(T168m.household_key, T168m.commodity_desc))
    sp['is_rep'] = [(h,c) in prev for h,c in zip(sp.household_key, sp.commodity_desc)]
    out['repeat_rate_84'] = wmean_by(sp.reset_index(), 'household_key', sp.is_rep.values, sp.sales_value.values).reindex(hh)
    out['private_share_84'] = wmean_by(M, 'household_key', (M.brand=='Private').astype(float), M.sales_value).reindex(hh)
    out['grocery_share_84'] = wmean_by(M, 'household_key', M.is_grocery, M.sales_value).reindex(hh)
    med_b = b84.groupby('household_key').sales.median()
    r = (b84.sales/(b84.index.get_level_values(0).map(med_b)+1e-9))
    out['stockup_max_84'] = r.groupby(b84.index.get_level_values(0)).max().reindex(hh)
    st = M.groupby(['household_key','store_id']).sales_value.sum().reset_index()
    out['store_loyalty_84'] = (st.groupby('household_key').sales_value.max()/st.groupby('household_key').sales_value.sum()).reindex(hh)
    T84['is_wkend'] = (((T84.day % 7)==5) | ((T84.day % 7)==6)).astype(float)
    out['wkend_share_84'] = wmean_by(T84.reset_index(), 'household_key', T84.is_wkend.values, T84.sales_value.values).reindex(hh)
    D = view.table('demographics')
    if D is not None and len(D):
        D = D.set_index('household_key')
        for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc']:
            out[c] = D[c].reindex(hh)
        out['has_demo'] = pd.Series(out.index.isin(D.index).astype(float), index=hh)
    out['snap_day'] = day; out['snap_week'] = (day+8)//7; out['snap_cycle_pos'] = day % 28
    return out

f4 = agent_api.build_features(feats_v4)
agent_api.save_table(f4, 'feats_v4.parquet')
print('saved', f4.shape)
new = [c for c in f4.columns if c not in agent_api.load_saved('feats_v3.parquet').columns]
print('new cols:', new)
print(f4[new].describe().T.round(3))


# ---- cell ----
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

def wmean_by(keys, val, w):
    keys = np.asarray(keys); ww = np.clip(np.asarray(w, float), 0, None); val = np.asarray(val, float)
    k = pd.Series(keys)
    num = pd.Series(val*ww).groupby(k).sum()
    den = pd.Series(ww).groupby(k).sum()+1e-9
    return num/den

def feats_v4(view, snapshot_day):
    T = view.table('transactions')
    P = view.table('products')
    hh = list(view.households)
    day = snapshot_day
    out = pd.DataFrame(index=hh)
    for w in [7,14,28,56,84,168]:
        out[f'spend_{w}'] = T[T.day > day-w].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['spend_all'] = T.groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    for k in [1,2,3]:
        lo, hi = day-28*k-27, day-28*k
        out[f'lag{k}_spend'] = T[(T.day>=lo)&(T.day<=hi)].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['exp4w_blend'] = 0.5*out.lag1_spend + 0.3*out.lag2_spend + 0.2*out.lag3_spend
    out['trend28'] = out.lag1_spend/(out.lag2_spend+5)
    out['trend84'] = out.spend_28/(out.spend_84/3+5)
    for k in range(8):
        out[f'wk{k}'] = T[(T.day>day-7*(k+1))&(T.day<=day-7*k)].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['active_weeks8'] = (out[[f'wk{k}' for k in range(8)]]>0).sum(1)
    out['wk_mean8'] = out[[f'wk{k}' for k in range(8)]].mean(1)
    out['wk_std8'] = out[[f'wk{k}' for k in range(8)]].std(1).fillna(0)
    out['wk_cv8'] = out.wk_std8/(out.wk_mean8+1)
    out['tenure'] = day - T.groupby('household_key').day.min().reindex(hh)
    out['days_since_last'] = day - T.groupby('household_key').day.max().reindex(hh)
    b = T.groupby(['household_key','basket_id']).agg(sales=('sales_value','sum'), d=('day','max'))
    b = b.sort_values(['household_key','d'])
    b84 = b[b.d > day-84]
    out['trips_84'] = b84.groupby('household_key').size().reindex(hh).fillna(0)
    out['trips_28'] = b[b.d>day-28].groupby('household_key').size().reindex(hh).fillna(0)
    out['basket_mean_84'] = b84.groupby('household_key').sales.mean().reindex(hh)
    out['basket_max_84'] = b84.groupby('household_key').sales.max().reindex(hh)
    out['basket_std_84'] = b84.groupby('household_key').sales.std().reindex(hh)
    out['spend_per_trip28'] = out.spend_28/(out.trips_28+1e-9)
    b['prev_d'] = b.groupby('household_key').d.shift(1)
    b['gap'] = b.d - b.prev_d
    g84 = b[(b.d>day-84) & b.gap.notna()]
    out['gap_mean_84'] = g84.groupby('household_key').gap.mean().reindex(hh)
    out['gap_max_84'] = g84.groupby('household_key').gap.max().reindex(hh)
    out['nprod_84'] = T[T.day>day-84].groupby('household_key').product_id.nunique().reindex(hh).fillna(0)
    out['nstore_84'] = T[T.day>day-84].groupby('household_key').store_id.nunique().reindex(hh).fillna(0)
    out['n_days_active_84'] = T[T.day>day-84].groupby('household_key').day.nunique().reindex(hh).fillna(0)
    out['coupon_disc_84'] = T[T.day>day-84].groupby('household_key').coupon_disc.sum().reindex(hh).fillna(0)
    out['retail_disc_84'] = T[T.day>day-84].groupby('household_key').retail_disc.sum().reindex(hh).fillna(0)
    out['coupon_match_disc_84'] = T[T.day>day-84].groupby('household_key').coupon_match_disc.sum().reindex(hh).fillna(0)
    out['disc_share_84'] = (out.coupon_disc_84+out.retail_disc_84+out.coupon_match_disc_84)/(out.spend_84+1e-9)
    T84 = T[T.day>day-84].copy()
    T84['morn'] = (T84.trans_time<1200).astype(float)
    out['morn_share_84'] = T84.groupby('household_key').morn.mean().reindex(hh)
    w_all = 0.97**((day-T.day).clip(lower=0))
    out['exp4w_all'] = (T.sales_value*w_all).groupby(T.household_key).sum().reindex(hh).fillna(0)
    T168 = T[T.day>day-168]
    w168 = 0.97**((day-T168.day).clip(lower=0))
    out['exp4w_168'] = (T168.sales_value*w168).groupby(T168.household_key).sum().reindex(hh).fillna(0)
    out['exp4w_84'] = out.spend_84
    out['weekly_rate_all'] = out.spend_all/(out.tenure+1)
    # ==== NEW v4 ====
    M = T84.merge(P[['product_id','commodity_desc','brand','department']], on='product_id', how='left')
    M['commodity_desc'] = M.commodity_desc.astype(object).where(M.commodity_desc.notna(), 'UNK')
    M['brand'] = M.brand.astype(object).where(M.brand.notna(), 'UNK')
    M['is_grocery'] = (M.department=='GROCERY').astype(float)
    sp = M.groupby(['household_key','commodity_desc']).sales_value.sum().reset_index()
    tot = sp.groupby('household_key').sales_value.transform('sum')
    sp['share'] = sp.sales_value/tot
    out['hhi_84'] = sp.assign(s2=sp.share**2).groupby('household_key').s2.sum().reindex(hh)
    out['n_commodities_84'] = sp.groupby('household_key').size().reindex(hh).fillna(0)
    out['top_comm_share_84'] = sp.groupby('household_key').share.max().reindex(hh)
    T168m = T[(T.day>day-168)&(T.day<=day-84)].merge(P[['product_id','commodity_desc']], on='product_id', how='left')
    T168m['commodity_desc'] = T168m.commodity_desc.astype(object).where(T168m.commodity_desc.notna(), 'UNK')
    prev = set(zip(T168m.household_key, T168m.commodity_desc))
    sp['is_rep'] = [(h,c) in prev for h,c in zip(sp.household_key, sp.commodity_desc)]
    out['repeat_rate_84'] = wmean_by(sp.household_key, sp.is_rep, sp.sales_value).reindex(hh)
    out['private_share_84'] = wmean_by(M.household_key, (M.brand=='Private').astype(float), M.sales_value).reindex(hh)
    out['grocery_share_84'] = wmean_by(M.household_key, M.is_grocery, M.sales_value).reindex(hh)
    med_b = b84.groupby('household_key').sales.median()
    r = (b84.sales/(b84.index.get_level_values(0).map(med_b)+1e-9))
    out['stockup_max_84'] = r.groupby(b84.index.get_level_values(0)).max().reindex(hh)
    st = M.groupby(['household_key','store_id']).sales_value.sum().reset_index()
    out['store_loyalty_84'] = (st.groupby('household_key').sales_value.max()/st.groupby('household_key').sales_value.sum()).reindex(hh)
    T84['is_wkend'] = (((T84.day % 7)==5) | ((T84.day % 7)==6)).astype(float)
    out['wkend_share_84'] = wmean_by(T84.household_key, T84.is_wkend, T84.sales_value).reindex(hh)
    D = view.table('demographics')
    if D is not None and len(D):
        D = D.set_index('household_key')
        for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc']:
            out[c] = D[c].reindex(hh)
        out['has_demo'] = pd.Series(out.index.isin(D.index).astype(float), index=hh)
    out['snap_day'] = day; out['snap_week'] = (day+8)//7; out['snap_cycle_pos'] = day % 28
    return out

f4 = agent_api.build_features(feats_v4)
agent_api.save_table(f4, 'feats_v4.parquet')
print('saved', f4.shape)
new = [c for c in f4.columns if c not in agent_api.load_saved('feats_v3.parquet').columns]
print('new cols:', new)
print(f4[new].describe().T.round(3))


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
f4 = agent_api.load_saved('feats_v4.parquet')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
DEMO = ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc']
def prep(f):
    m = tt.merge(f, on=['household_key','snapshot_day'], how='left')
    drop = ['index','household_key','snapshot_day','future_spend_4w']
    feats = [c for c in m.columns if c not in drop and c not in DEMO]
    m[feats] = m[feats].fillna(-1)
    for c in DEMO:
        m[c] = m[c].astype('category').cat.codes
    return m, feats + DEMO
m4, F4 = prep(f4)
m3, F3 = prep(f3)
params = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=4, min_child_weight=20,
              learning_rate=0.05, n_estimators=800, subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=4)
# quick clean CV check: v4 vs v3 (pure model)
for name, m, F in [('v3', m3, F3), ('v4', m4, F4)]:
    tr = m[m.snapshot_day <= 347]; te = m[m.snapshot_day.isin([375,431])]
    mod = xgb.XGBRegressor(**params).fit(tr[F], tr.future_spend_4w)
    pr = np.clip(mod.predict(te[F]), 0, None)
    maes = [round(float(np.abs(pr[te.snapshot_day==s]-te[te.snapshot_day==s].future_spend_4w.values).mean()),3) for s in [375,431]]
    print(name, 'CV 375/431:', maes, round(np.mean(maes),3))
# final: train on all train snaps, predict validation
tr = m4[m4.snapshot_day <= 431]
mod = xgb.XGBRegressor(**params).fit(tr[F4], tr.future_spend_4w)
val = m4[m4.snapshot_day.isin([459,487,515,543])].copy()
val['prediction'] = np.clip(mod.predict(val[F4]), 0, None)
out = val[['household_key','snapshot_day','prediction']]
agent_api.save_table(out, 'pred_e008.parquet')
print('pred rows', out.shape, out.snapshot_day.value_counts().to_dict())
