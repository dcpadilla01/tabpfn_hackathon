import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
print('KEYS', agent_api.KEYS, 'TARGET', agent_api.TARGET)
print('snapshot_days', agent_api.snapshot_days())
base = agent_api.load_saved('e013_stationary.parquet')
print('base shape', base.shape)
print('base cols:', sorted(base.columns))
s = agent_api.snapshot()
print('households type', type(s.households), 'len', len(s.households))
tx = s.table('transactions')
print('tx shape', tx.shape)
tt = agent_api.train_targets()
print('targets shape', tt.shape)
print(tt.future_spend_4w.describe())
print('zero share', (tt.future_spend_4w==0).mean())


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
s = agent_api.snapshot()
print('households repr:', repr(s.households)[:200])
print('day', s.day, 'week', s.week)
tx = s.table('transactions')
print('tx shape', tx.shape)
print(tx.head(3))
print('n hh', tx.household_key.nunique())
# check a household history
h = agent_api.history(tx.household_key.iloc[0])
print('hist shape', h.shape)
print(h.head(3))


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
tt = agent_api.train_targets()
print('targets shape', tt.shape)
print(tt.future_spend_4w.describe())
print('zero share', (tt.future_spend_4w==0).mean())
# How predictable is future spend from lag1 (spend in [s-27,s])? quick sanity on train rows
base = agent_api.load_saved('e013_stationary.parquet')
print('base shape', base.shape)
m = tt.merge(base[['household_key','snapshot_day','spend_28','spend_84','z_rate84','dec_28']], on=['household_key','snapshot_day'])
print('merged', m.shape)
for c in ['spend_28','spend_84','z_rate84','dec_28']:
    print(c, 'corr', np.corrcoef(m[c].fillna(0), m.future_spend_4w)[0,1])


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
# Correlations of E013 features with target on train rows
tt = agent_api.train_targets()
base = agent_api.load_saved('e013_stationary.parquet')
m = tt.merge(base, on=['household_key','snapshot_day'])
feats = [c for c in base.columns if c not in ('household_key','snapshot_day')]
rows=[]
for c in feats:
    x = m[c]
    if x.dtype.kind not in 'biufc':
        continue
    ok = x.notna()
    if ok.sum() < 100: 
        rows.append((c, np.nan, ok.sum())); continue
    rows.append((c, np.corrcoef(x[ok], m.future_spend_4w[ok])[0,1], ok.sum()))
corr = pd.DataFrame(rows, columns=['feat','corr','n']).sort_values('corr', key=lambda s: s.abs(), ascending=False)
print(corr.head(30).to_string())
print(corr.tail(15).to_string())


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
s = agent_api.snapshot()
tx = s.table('transactions')
# Market-level trend: avg weekly spend per active household, and avg unit price by week
tx['week'] = (tx.day+8)//7
wk = tx.groupby('week').agg(total=('sales_value','sum'), hh=('household_key','nunique'), lines=('sales_value','size'))
wk['per_hh'] = wk.total/wk.hh
print(wk[['per_hh','lines']].describe())
print(wk.iloc[::6][['per_hh']].T.to_string())
# price proxy: sales_value per line over time (monthly)
tx['m'] = tx.day//28
mp = tx.groupby('m').sales_value.mean()
print('mean line value by 28d month:'); print(mp.iloc[::3].round(3).to_string())
# week alignment check
for d in [95,123,459,487,515,543]:
    print(d, 'week', (d+8)//7, 'mod52', ((d+8)//7-1)%52+1)
# max day in data
print('max day', tx.day.max())


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
# Build candidate features inside build_features: market seasonal level, holiday flags, price level, discount mix, dept mix extra
def fn(view, snapshot_day):
    tx = view.table('transactions')
    tx = tx[tx.sales_value.notna()]
    week = (snapshot_day+8)//7
    tx = tx.assign(w=(tx.day+8)//7)
    out = pd.DataFrame(index=view.households.index if view.households is not None else pd.Index([]))
    return out
# first check what view.households is inside build_features at a train snapshot
def fn2(view, snapshot_day):
    print('inside: day', view.day, 'week', view.week, 'households', type(view.households))
    tx = view.table('transactions')
    print('tx max day', tx.day.max(), 'shape', tx.shape)
    return pd.DataFrame(index=pd.Index([], name='household_key'))
import agent_api
bf = agent_api.build_features(fn2)
print(bf.shape)


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
s = agent_api.snapshot()
tx = s.table('transactions')
prod = s.table('products')
m = tx.merge(prod[['product_id','department']], on='product_id', how='left')
print(m.department.value_counts().head(25).to_string())
# discount columns sign
print(tx[['sales_value','coupon_disc','coupon_match_disc','retail_disc']].describe().loc[['mean','min','max']].to_string())


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
# Precompute static-ish lookups for speed: dept map, and check timing of groupby ops
s = agent_api.snapshot()
tx = s.table('transactions')
prod = s.table('products')
dept = prod.set_index('product_id').department
t0=time.time() if False else None
import time as _t
t0=_t.time()
tx['w']=(tx.day+8)//7
g = tx.groupby(['household_key','w']).sales_value.sum()
print('hh-week groupby time', round(_t.time()-t0,2))
t0=_t.time()
tx['dept'] = tx.product_id.map(dept)
g2 = tx.groupby(['household_key','dept']).sales_value.sum()
print('hh-dept groupby time', round(_t.time()-t0,2), g2.shape)
t0=_t.time()
g3 = tx.groupby(['household_key','w','dept']).sales_value.sum()
print('hh-week-dept time', round(_t.time()-t0,2), g3.shape)


# ---- cell ----
import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore')
# Prototype E014 fn and time it at the largest snapshot
def fn(view, snapshot_day):
    tx = view.table('transactions')
    tx = tx[tx.sales_value.notna()].copy()
    tx['w'] = (tx.day + 8)//7
    wk = tx.groupby('w').sales_value.sum()
    wk = wk.reindex(range(1, week_max+1)).ffill()
    lvl = wk/wk.iloc[:8].mean()
    hh = view.households
    g = tx.groupby(['household_key','w']).sales_value.sum()
    idx = pd.MultiIndex.from_product([hh, range(max(1,week-25), week+1)], names=['household_key','w'])
    g = g.reindex(idx).fillna(0.0)
    z = g.xs(hh, level=0) if False else None
    arr = np.zeros((len(hh), 26))
    for i,h in enumerate(hh):
        arr[i] = g.xs(h, level=0).values
    return None
week_max = 70
import agent_api, time
t0=time.time()
# skip prototype; instead time the reindex approach on full tx at day 459
s = agent_api.snapshot()
tx = s.table('transactions')
tx = tx[tx.sales_value.notna()].copy()
tx['w'] = (tx.day+8)//7
g = tx.groupby(['household_key','w']).sales_value.sum()
hh = tx.household_key.unique()
t0=time.time()
idx = pd.MultiIndex.from_product([hh, range(41, 67)], names=['household_key','w'])
gg = g.reindex(idx).fillna(0.0)
mat = gg.unstack()  # hh x weeks
print('unstack time', round(time.time()-t0,2), mat.shape)
t0=time.time()
roll = mat.rolling(4, axis=1).mean()
print('rolling time', round(time.time()-t0,2))
# holiday flags
wmod = np.array([( (w-1)%52 )+1 for w in range(41,67)])
hol = ((wmod>=51)|(wmod<=1)).astype(float)
print('holiday weeks in window:', hol)


# ---- cell ----
import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore')
# Full E014 prototype: build features at snapshot 459 and check correlations with train targets at 459
import agent_api
def build(view, snapshot_day):
    tx = view.table('transactions')
    tx = tx[tx.sales_value.notna()].copy()
    tx['w'] = (tx.day + 8)//7
    week = (snapshot_day + 8)//7
    # ---- market level (seasonal) ----
    wk_tot = tx.groupby('w').sales_value.sum()
    wk_tot = wk_tot.reindex(range(1, week+1)).ffill()
    lvl = wk_tot / wk_tot.iloc[:8].mean()
    # ---- hh-week spend matrix, last 26 weeks ----
    g = tx.groupby(['household_key','w']).sales_value.sum()
    weeks = range(max(1, week-25), week+1)
    idx = pd.MultiIndex.from_product([view.households, weeks], names=['household_key','w'])
    mat = g.reindex(idx).fillna(0.0).unstack()   # hh x 26 weeks
    mat = mat.reindex(view.households)
    # seasonal features
    f = pd.DataFrame(index=view.households)
    f['mkt_lvl'] = lvl.loc[weeks].mean()
    f['mkt_lvl_last4'] = lvl.loc[list(weeks)[-4:]].mean()
    f['mkt_hol_ahead'] = float(((week%52==0) or ((week+1)%52==1)))
    f['mkt_hol_now'] = float(week%52==0)
    # household seasonal shape
    wmod = np.array([((w-1)%52)+1 for w in weeks])
    hol = ((wmod>=51)|(wmod<=1)).astype(float)
    f['hh_hol_wk_share'] = (mat.values*hol).sum(1)/max(1e-9, mat.values.sum(1))
    f['hh_hol_wk_amt'] = (mat.values*hol).sum(1)
    # rolling 4-week sums (aligned with future window)
    r = mat.rolling(4, axis=1).mean().values*4
    f['r4_lag1'] = r[:,-1]      # [week-3..week]
    f['r4_lag2'] = r[:,-2]      # [week-4..week-1]
    f['r4_lag3'] = r[:,-3]
    f['r4_lag4'] = r[:,-4]
    f['r4_lag5'] = r[:,-5]
    f['r4_lag13'] = r[:,-13]
    f['r4_lag14'] = r[:,-14]
    f['r4_lag26'] = r[:,-26]
    f['r4_ratio_l2_l14'] = r[:,-2]/(r[:,-14]+1)
    f['r4_ratio_l1_l13'] = r[:,-1]/(r[:,-13]+1)
    f['r4_ratio_l13_l26'] = r[:,-13]/(r[:,-26]+1)
    f['r4_mean_l1_l26'] = r.mean(1)
    # weekly series stats
    f['wk_max'] = mat.values.max(1)
    f['wk_min'] = mat.values.min(1)
    f['wk_std'] = mat.values.std(1)
    f['wk_last'] = mat.values[:,-1]
    f['wk_active'] = (mat.values>0).sum(1)
    # ---- discount mix (84d) ----
    d0 = snapshot_day-83
    txd = tx[(tx.day>=d0)&(tx.day<=snapshot_day)]
    agg = txd.groupby('household_key').agg(sales=('sales_value','sum'),
        coup=('coupon_disc','sum'), match=('coupon_match_disc','sum'), retail=('retail_disc','sum'),
        nlines=('sales_value','size'))
    agg = agg.reindex(view.households).fillna(0.0)
    f['disc_retail_84'] = -agg.retail
    f['disc_coup_84'] = -agg.coup
    f['disc_match_84'] = -agg.match
    f['disc_total_84'] = -(agg.retail+agg.coup+agg.match)
    f['disc_share_84'] = (-(agg.retail+agg.coup+agg.match))/(agg.sales+1)
    f['lines_84'] = agg.nlines
    f['lines_per_dollar_84'] = agg.nlines/(agg.sales+1)
    # ---- dept mix 84d ----
    prod = view.table('products')
    dept = prod.set_index('product_id').department
    txd = txd.assign(dept=txd.product_id.map(dept))
    gd = txd.groupby(['household_key','dept']).sales_value.sum().unstack().reindex(view.households).fillna(0.0)
    tot = gd.sum(1)
    for c in ['GROCERY','PRODUCE','MEAT','MEAT-PCKGD','DRUG GM','DELI','PASTRY','KIOSK-GAS','NUTRITION','SEAFOOD-PCKGD','SALAD BAR','COSMETICS','MISC SALES TRAN','FLORAL','SPIRITS','SEAFOOD']:
        if c in gd.columns:
            f['d84_'+c] = gd[c]/(tot+1)
        else:
            f['d84_'+c] = 0.0
    f['d84_n_depts'] = (gd>0).sum(1)
    f['d84_entropy'] = -(np.where(gd.values>0, gd.values/(tot.values[:,None]+1e-9), 0)*np.log((gd.values/(tot.values[:,None]+1e-9)).clip(1e-9))).sum(1)
    return f

t0=time.time()
bf = agent_api.build_features(build)
print('build time', round(time.time()-t0,1), bf.shape)
tt = agent_api.train_targets()
m = tt.merge(bf, on=['household_key','snapshot_day'])
print('merged', m.shape)
res=[]
for c in bf.columns:
    x=m[c]; ok=x.notna()
    res.append((c, np.corrcoef(x[ok], m.future_spend_4w[ok])[0,1], ok.mean()))
res=pd.DataFrame(res, columns=['feat','corr','n']).sort_values('corr', key=lambda s: s.abs(), ascending=False)
print(res.head(35).to_string())
print(res.tail(15).to_string())


# ---- cell ----
import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore')
import agent_api
def build(view, snapshot_day):
    tx = view.table('transactions')
    tx = tx[tx.sales_value.notna()].copy()
    tx['w'] = (tx.day + 8)//7
    week = (snapshot_day + 8)//7
    wk_tot = tx.groupby('w').sales_value.sum()
    wk_tot = wk_tot.reindex(range(1, week+1)).ffill()
    lvl = wk_tot / wk_tot.iloc[:8].mean()
    g = tx.groupby(['household_key','w']).sales_value.sum()
    weeks = list(range(max(1, week-25), week+1))
    idx = pd.MultiIndex.from_product([view.households, weeks], names=['household_key','w'])
    mat = g.reindex(idx).fillna(0.0).unstack()
    mat = mat.reindex(view.households)
    f = pd.DataFrame(index=view.households)
    f['mkt_lvl'] = float(lvl.loc[weeks].mean())
    f['mkt_lvl_last4'] = float(lvl.loc[weeks[-4:]].mean())
    f['mkt_hol_ahead'] = float((week%52==0) or ((week+1)%52==1))
    f['mkt_hol_now'] = float(week%52==0)
    wmod = np.array([((w-1)%52)+1 for w in weeks])
    hol = ((wmod>=51)|(wmod<=1)).astype(float)
    V = mat.values
    f['hh_hol_wk_share'] = (V*hol).sum(1)/np.maximum(1e-9, V.sum(1))
    f['hh_hol_wk_amt'] = (V*hol).sum(1)
    r = mat.rolling(4, axis=1).mean().values*4
    f['r4_lag1'] = r[:,-1]
    f['r4_lag2'] = r[:,-2]
    f['r4_lag3'] = r[:,-3]
    f['r4_lag4'] = r[:,-4]
    f['r4_lag5'] = r[:,-5]
    f['r4_lag13'] = r[:,-13]
    f['r4_lag14'] = r[:,-14]
    f['r4_lag26'] = r[:,-26]
    f['r4_ratio_l2_l14'] = r[:,-2]/(r[:,-14]+1)
    f['r4_ratio_l1_l13'] = r[:,-1]/(r[:,-13]+1)
    f['r4_ratio_l13_l26'] = r[:,-13]/(r[:,-26]+1)
    f['r4_mean_l1_l26'] = r.mean(1)
    f['wk_max'] = V.max(1)
    f['wk_min'] = V.min(1)
    f['wk_std'] = V.std(1)
    f['wk_last'] = V[:,-1]
    f['wk_active'] = (V>0).sum(1)
    d0 = snapshot_day-83
    txd = tx[(tx.day>=d0)&(tx.day<=snapshot_day)]
    agg = txd.groupby('household_key').agg(sales=('sales_value','sum'),
        coup=('coupon_disc','sum'), match=('coupon_match_disc','sum'), retail=('retail_disc','sum'),
        nlines=('sales_value','size'))
    agg = agg.reindex(view.households).fillna(0.0)
    f['disc_retail_84'] = -agg.retail
    f['disc_coup_84'] = -agg.coup
    f['disc_match_84'] = -agg.match
    f['disc_total_84'] = -(agg.retail+agg.coup+agg.match)
    f['disc_share_84'] = (-(agg.retail+agg.coup+agg.match))/(agg.sales+1)
    f['lines_84'] = agg.nlines
    f['lines_per_dollar_84'] = agg.nlines/(agg.sales+1)
    prod = view.table('products')
    dept = prod.set_index('product_id').department
    txd = txd.assign(dept=txd.product_id.map(dept))
    gd = txd.groupby(['household_key','dept']).sales_value.sum().unstack().reindex(view.households).fillna(0.0)
    tot = gd.sum(1)
    for c in ['GROCERY','PRODUCE','MEAT','MEAT-PCKGD','DRUG GM','DELI','PASTRY','KIOSK-GAS','NUTRITION','SEAFOOD-PCKGD','SALAD BAR','COSMETICS','MISC SALES TRAN','FLORAL','SPIRITS','SEAFOOD']:
        f['d84_'+c] = gd[c]/(tot+1) if c in gd.columns else 0.0
    f['d84_n_depts'] = (gd>0).sum(1)
    p = (gd.values/(tot.values[:,None]+1e-9))
    f['d84_entropy'] = -(np.where(gd.values>0, p, 0)*np.log(p.clip(1e-9))).sum(1)
    return f

t0=time.time()
bf = agent_api.build_features(build)
print('build time', round(time.time()-t0,1), bf.shape)
tt = agent_api.train_targets()
m = tt.merge(bf, on=['household_key','snapshot_day'])
print('merged', m.shape)
res=[]
for c in bf.columns:
    x=m[c]; ok=x.notna()
    res.append((c, np.corrcoef(x[ok], m.future_spend_4w[ok])[0,1], ok.mean()))
res=pd.DataFrame(res, columns=['feat','corr','n']).sort_values('corr', key=lambda s: s.abs(), ascending=False)
print(res.head(30).to_string())
print(res.tail(18).to_string())
path = agent_api.save_table(bf, 'e014_seasonal_proto.parquet')
print('saved', path)


# ---- cell ----
import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore')
import agent_api
def build(view, snapshot_day):
    tx = view.table('transactions')
    tx = tx[tx.sales_value.notna()].copy()
    tx['w'] = (tx.day + 8)//7
    week = (snapshot_day + 8)//7
    wk_tot = tx.groupby('w').sales_value.sum()
    wk_tot = wk_tot.reindex(range(1, week+1)).ffill()
    lvl = wk_tot / wk_tot.iloc[:8].mean()
    g = tx.groupby(['household_key','w']).sales_value.sum()
    weeks = list(range(max(1, week-25), week+1))
    idx = pd.MultiIndex.from_product([view.households, weeks], names=['household_key','w'])
    mat = g.reindex(idx).fillna(0.0).unstack()
    mat = mat.reindex(view.households)
    # left-pad with NaN so lag features are NaN when window doesn't exist yet
    if mat.shape[1] < 26:
        pad = pd.DataFrame(np.nan, index=mat.index, columns=range(1, 27-mat.shape[1]))
        mat = pd.concat([pad, mat], axis=1)
    f = pd.DataFrame(index=view.households)
    f['mkt_lvl'] = float(lvl.loc[weeks].mean())
    f['mkt_lvl_last4'] = float(lvl.loc[weeks[-4:]].mean())
    f['mkt_hol_ahead'] = float((week%52==0) or ((week+1)%52==1))
    f['mkt_hol_now'] = float(week%52==0)
    wmod = np.array([((w-1)%52)+1 for w in range(week-25, week+1) if w>=1])
    hol = ((wmod>=51)|(wmod<=1)).astype(float)
    V = np.nan_to_num(mat.values)
    f['hh_hol_wk_share'] = (V[-len(hol):]*hol).sum(1)/np.maximum(1e-9, V.sum(1))
    f['hh_hol_wk_amt'] = (V[-len(hol):]*hol).sum(1)
    r = mat.rolling(4, axis=1, min_periods=4).mean().values*4
    f['r4_lag1'] = r[:,-1]; f['r4_lag2'] = r[:,-2]; f['r4_lag3'] = r[:,-3]
    f['r4_lag4'] = r[:,-4]; f['r4_lag5'] = r[:,-5]
    f['r4_lag13'] = r[:,-13]; f['r4_lag14'] = r[:,-14]; f['r4_lag26'] = r[:,-26]
    with np.errstate(all='ignore'):
        f['r4_ratio_l2_l14'] = r[:,-2]/(r[:,-14]+1)
        f['r4_ratio_l1_l13'] = r[:,-1]/(r[:,-13]+1)
        f['r4_ratio_l13_l26'] = r[:,-13]/(r[:,-26]+1)
    f['r4_mean_l1_l26'] = np.nanmean(r, axis=1)
    f['wk_max'] = np.nanmax(mat.values, axis=1)
    f['wk_std'] = np.nanstd(mat.values, axis=1)
    f['wk_last'] = mat.values[:,-1]
    f['wk_active'] = (np.nan_to_num(mat.values)>0).sum(1)
    d0 = snapshot_day-83
    txd = tx[(tx.day>=d0)&(tx.day<=snapshot_day)]
    agg = txd.groupby('household_key').agg(sales=('sales_value','sum'),
        coup=('coupon_disc','sum'), match=('coupon_match_disc','sum'), retail=('retail_disc','sum'),
        nlines=('sales_value','size'))
    agg = agg.reindex(view.households).fillna(0.0)
    f['disc_retail_84'] = -agg.retail
    f['disc_coup_84'] = -agg.coup
    f['disc_match_84'] = -agg.match
    f['disc_total_84'] = -(agg.retail+agg.coup+agg.match)
    f['disc_share_84'] = (-(agg.retail+agg.coup+agg.match))/(agg.sales+1)
    f['lines_84'] = agg.nlines
    f['lines_per_dollar_84'] = agg.nlines/(agg.sales+1)
    prod = view.table('products')
    dept = prod.set_index('product_id').department
    txd = txd.assign(dept=txd.product_id.map(dept))
    gd = txd.groupby(['household_key','dept']).sales_value.sum().unstack().reindex(view.households).fillna(0.0)
    tot = gd.sum(1)
    for c in ['GROCERY','PRODUCE','MEAT','MEAT-PCKGD','DRUG GM','DELI','PASTRY','KIOSK-GAS','NUTRITION','SEAFOOD-PCKGD','SALAD BAR','COSMETICS','MISC SALES TRAN','FLORAL','SPIRITS','SEAFOOD']:
        f['d84_'+c] = gd[c]/(tot+1) if c in gd.columns else 0.0
    f['d84_n_depts'] = (gd>0).sum(1)
    p = (gd.values/(tot.values[:,None]+1e-9))
    f['d84_entropy'] = -(np.where(gd.values>0, p, 0)*np.log(p.clip(1e-9))).sum(1)
    return f

t0=time.time()
bf = agent_api.build_features(build)
print('build time', round(time.time()-t0,1), bf.shape)
tt = agent_api.train_targets()
m = tt.merge(bf, on=['household_key','snapshot_day'])
print('merged', m.shape)
res=[]
for c in bf.columns:
    x=m[c]; ok=x.notna()
    res.append((c, np.corrcoef(x[ok], m.future_spend_4w[ok])[0,1], ok.mean()))
res=pd.DataFrame(res, columns=['feat','corr','n']).sort_values('corr', key=lambda s: s.abs(), ascending=False)
print(res.head(28).to_string())
print(res.tail(12).to_string())
path = agent_api.save_table(bf, 'e014_seasonal.parquet')
print('saved', path)


# ---- cell ----
import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore')
import agent_api
def build(view, snapshot_day):
    tx = view.table('transactions')
    tx = tx[tx.sales_value.notna()].copy()
    tx['w'] = (tx.day + 8)//7
    week = (snapshot_day + 8)//7
    wk_tot = tx.groupby('w').sales_value.sum()
    wk_tot = wk_tot.reindex(range(1, week+1)).ffill()
    lvl = wk_tot / wk_tot.iloc[:8].mean()
    g = tx.groupby(['household_key','w']).sales_value.sum()
    weeks = list(range(max(1, week-25), week+1))
    idx = pd.MultiIndex.from_product([view.households, weeks], names=['household_key','w'])
    mat = g.reindex(idx).fillna(0.0).unstack()
    mat = mat.reindex(view.households)
    if mat.shape[1] < 26:
        pad = pd.DataFrame(np.nan, index=mat.index, columns=range(1, 27-mat.shape[1]))
        mat = pd.concat([pad, mat], axis=1)
    f = pd.DataFrame(index=view.households)
    f['mkt_lvl'] = float(lvl.loc[weeks].mean())
    f['mkt_lvl_last4'] = float(lvl.loc[weeks[-4:]].mean())
    f['mkt_hol_ahead'] = float((week%52==0) or ((week+1)%52==1))
    f['mkt_hol_now'] = float(week%52==0)
    # holiday weights aligned to the 26 padded columns (w = week-25 .. week; 0 if w<1)
    warr = np.array([w if w>=1 else 0 for w in range(week-25, week+1)])
    wmod = np.where(warr>0, ((warr-1)%52)+1, 0)
    hol = ((wmod>=51)|(wmod==1)).astype(float)
    V = np.nan_to_num(mat.values)
    f['hh_hol_wk_share'] = (V*hol).sum(1)/np.maximum(1e-9, V.sum(1))
    f['hh_hol_wk_amt'] = (V*hol).sum(1)
    r = mat.rolling(4, axis=1, min_periods=4).mean().values*4
    f['r4_lag1'] = r[:,-1]; f['r4_lag2'] = r[:,-2]; f['r4_lag3'] = r[:,-3]
    f['r4_lag4'] = r[:,-4]; f['r4_lag5'] = r[:,-5]
    f['r4_lag13'] = r[:,-13]; f['r4_lag14'] = r[:,-14]; f['r4_lag26'] = r[:,-26]
    with np.errstate(all='ignore'):
        f['r4_ratio_l2_l14'] = r[:,-2]/(r[:,-14]+1)
        f['r4_ratio_l1_l13'] = r[:,-1]/(r[:,-13]+1)
        f['r4_ratio_l13_l26'] = r[:,-13]/(r[:,-26]+1)
    f['r4_mean_l1_l26'] = np.nanmean(r, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f['wk_max'] = np.nanmax(mat.values, axis=1)
        f['wk_std'] = np.nanstd(mat.values, axis=1)
    f['wk_last'] = mat.values[:,-1]
    f['wk_active'] = (np.nan_to_num(mat.values)>0).sum(1)
    d0 = snapshot_day-83
    txd = tx[(tx.day>=d0)&(tx.day<=snapshot_day)]
    agg = txd.groupby('household_key').agg(sales=('sales_value','sum'),
        coup=('coupon_disc','sum'), match=('coupon_match_disc','sum'), retail=('retail_disc','sum'),
        nlines=('sales_value','size'))
    agg = agg.reindex(view.households).fillna(0.0)
    f['disc_retail_84'] = -agg.retail
    f['disc_coup_84'] = -agg.coup
    f['disc_match_84'] = -agg.match
    f['disc_total_84'] = -(agg.retail+agg.coup+agg.match)
    f['disc_share_84'] = (-(agg.retail+agg.coup+agg.match))/(agg.sales+1)
    f['lines_84'] = agg.nlines
    f['lines_per_dollar_84'] = agg.nlines/(agg.sales+1)
    prod = view.table('products')
    dept = prod.set_index('product_id').department
    txd = txd.assign(dept=txd.product_id.map(dept))
    gd = txd.groupby(['household_key','dept']).sales_value.sum().unstack().reindex(view.households).fillna(0.0)
    tot = gd.sum(1)
    for c in ['GROCERY','PRODUCE','MEAT','MEAT-PCKGD','DRUG GM','DELI','PASTRY','KIOSK-GAS','NUTRITION','SEAFOOD-PCKGD','SALAD BAR','COSMETICS','MISC SALES TRAN','FLORAL','SPIRITS','SEAFOOD']:
        f['d84_'+c] = gd[c]/(tot+1) if c in gd.columns else 0.0
    f['d84_n_depts'] = (gd>0).sum(1)
    p = (gd.values/(tot.values[:,None]+1e-9))
    f['d84_entropy'] = -(np.where(gd.values>0, p, 0)*np.log(p.clip(1e-9))).sum(1)
    return f

t0=time.time()
bf = agent_api.build_features(build)
print('build time', round(time.time()-t0,1), bf.shape)
tt = agent_api.train_targets()
m = tt.merge(bf, on=['household_key','snapshot_day'])
print('merged', m.shape)
res=[]
for c in bf.columns:
    x=m[c]; ok=x.notna()
    res.append((c, np.corrcoef(x[ok], m.future_spend_4w[ok])[0,1], ok.mean()))
res=pd.DataFrame(res, columns=['feat','corr','n']).sort_values('corr', key=lambda s: s.abs(), ascending=False)
print(res.head(28).to_string())
print(res.tail(12).to_string())
path = agent_api.save_table(bf, 'e014_seasonal.parquet')
print('saved', path)
