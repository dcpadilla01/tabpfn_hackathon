import agent_api as A, pandas as pd, numpy as np

dm = A.load_saved('dm_exp.parquet')
print('dm_exp:', dm.shape)
print(dm.columns.tolist())
print(dm.head(3).to_string())
if 'snapshot_day' in dm.columns:
    print(dm['snapshot_day'].value_counts().sort_index())

e11 = A.load_saved('e011_price.parquet')
print('\ne011:', e11.shape, 'snapdays:', sorted(e11.snapshot_day.unique()))
print(e11.columns.tolist())

# retailer-wide weekly spend trend (view capped at 459 - fine for exploration)
v = A.snapshot()
t = v.transactions
w = t.groupby('week_no')['sales_value'].sum()
print('\nweekly retailer spend, last 12 weeks:')
print(w.tail(12).to_string())
y = w.values
print('mean weekly spend:', y.mean(), 'first13 mean:', y[:13].mean(), 'last13 mean:', y[-13:].mean())

# train target level by snapshot (train only)
tt = A.train_targets()
print('\ntrain target by snapshot:')
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count']).to_string())


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

e11 = A.load_saved('e011_price.parquet')
dm = A.load_saved('dm_exp.parquet')
print('e11', e11.shape, 'dm', dm.shape)
print('dm dup keys:', dm.duplicated(['household_key','snapshot_day']).sum(),
      '| e11 dup keys:', e11.duplicated(['household_key','snapshot_day']).sum())
dm2 = dm.copy()
dm2['household_key'] = dm2['household_key'].astype(e11['household_key'].dtype)
dm2['snapshot_day'] = dm2['snapshot_day'].astype(e11['snapshot_day'].dtype)
feat = [c for c in dm2.columns if c not in ('household_key','snapshot_day')]
m = e11.merge(dm2[['household_key','snapshot_day']+feat], on=['household_key','snapshot_day'], how='left')
print('merged:', m.shape, '| NaN dm_disp_spend28:', m['dm_disp_spend28'].isna().sum())
A.save_table(m, 'e014_dm.parquet')

v = A.snapshot()
h = v.households
if isinstance(h, pd.DataFrame):
    print('households DataFrame, cols:', h.columns.tolist(), 'shape:', h.shape)
    print(h.head(2).to_string())
else:
    print('households not DataFrame; first 5:', list(h)[:5])

def macro_fn(view, day):
    t = view.transactions
    d = t.groupby('day')['sales_value'].sum()
    hh = t.groupby('day')['household_key'].nunique()
    def wsum(a,b):
        i = d.index[(d.index>=a)&(d.index<=b)]
        return float(d.reindex(i).sum()) if len(i) else np.nan
    def whh(a,b):
        i = hh.index[(hh.index>=a)&(hh.index<=b)]
        return float(hh.reindex(i).sum()) if len(i) else np.nan
    h = view.households
    if isinstance(h, pd.DataFrame):
        idx = pd.Index(h['household_key'] if 'household_key' in h.columns else h.index)
    else:
        idx = pd.Index(list(h))
    r = {}
    r['macro_spend28'] = wsum(day-27, day)
    r['macro_spend_p28'] = wsum(day-55, day-28)
    r['macro_spend28_ly'] = wsum(day-391, day-364) if day >= 392 else np.nan
    r['macro_spend112'] = wsum(day-111, day)
    r['macro_spend112_ly'] = wsum(day-475, day-364) if day >= 476 else np.nan
    r['macro_hh28'] = whh(day-27, day)
    r['macro_spend_per_hh28'] = r['macro_spend28']/r['macro_hh28'] if r['macro_hh28'] else np.nan
    ly_hh = whh(day-391, day-364) if day>=392 else np.nan
    r['macro_spend_per_hh28_ly'] = r['macro_spend28_ly']/ly_hh if ly_hh else np.nan
    r['macro_ratio_ly'] = r['macro_spend28']/r['macro_spend28_ly'] if day>=392 else np.nan
    r['macro_growth'] = r['macro_spend28']/r['macro_spend_p28'] if r['macro_spend_p28'] else np.nan
    r['macro_ratio112_ly'] = r['macro_spend112']/r['macro_spend112_ly'] if day>=476 else np.nan
    wk = t.groupby('week_no')['sales_value'].sum()
    wcur = (day+8)//7
    l12 = wk.reindex(range(wcur-11, wcur+1)).values.astype(float)
    r['macro_wk_ratio_last3'] = np.nanmean(l12[-3:])/np.nanmean(l12[:-3]) if np.nanmean(l12[:-3]) else np.nan
    return pd.DataFrame({k: r[k] for k in r}, index=idx).astype(float)

mac = A.build_features(macro_fn)
print('macro table:', mac.shape, mac.columns.tolist())
ps = mac.drop_duplicates('snapshot_day').set_index('snapshot_day')
print(ps[['macro_spend28','macro_spend_per_hh28','macro_ratio_ly','macro_growth','macro_wk_ratio_last3']].round(3).to_string())
A.save_table(mac, 'macro.parquet')
print('saved macro + e014_dm')


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
v = A.snapshot()
print('households:', v.households)
print('day:', v.day, 'week:', v.week)
# maybe households derivable from transactions at this snapshot
t = v.transactions
print('txn shape:', t.shape, 'max day:', t['day'].max())
hh = t.groupby('household_key')['day'].min()
print('n households ever:', len(hh))
# eligibility rule: first purchase >= 84 days before snapshot
elig = hh[hh <= v.day - 84]
print('eligible at 459:', len(elig))
# compare with a saved table's row count per snapshot
e11 = A.load_saved('e011_price.parquet')
print('e11 rows at 459:', (e11.snapshot_day==459).sum())
# are the eligible keys exactly the e11 keys at 459?
e11h = set(e11[e11.snapshot_day==459]['household_key'])
print('match:', set(elig.index.astype(e11.household_key.dtype)) == e11h)


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

def macro_fn(view, day):
    t = view.transactions
    d = t.groupby('day')['sales_value'].sum()
    hh = t.groupby('day')['household_key'].nunique()
    first = t.groupby('household_key')['day'].min()
    idx = first.index[first <= day-84]
    def wsum(a,b):
        i = d.index[(d.index>=a)&(d.index<=b)]
        return float(d.reindex(i).sum()) if len(i) else np.nan
    def whh(a,b):
        i = hh.index[(hh.index>=a)&(hh.index<=b)]
        return float(hh.reindex(i).sum()) if len(i) else np.nan
    r = {}
    r['macro_spend28'] = wsum(day-27, day)
    r['macro_spend_p28'] = wsum(day-55, day-28)
    r['macro_spend28_ly'] = wsum(day-391, day-364) if day >= 392 else np.nan
    r['macro_spend112'] = wsum(day-111, day)
    r['macro_spend112_ly'] = wsum(day-475, day-364) if day >= 476 else np.nan
    r['macro_hh28'] = whh(day-27, day)
    r['macro_spend_per_hh28'] = r['macro_spend28']/r['macro_hh28'] if r['macro_hh28'] else np.nan
    ly_hh = whh(day-391, day-364) if day>=392 else np.nan
    r['macro_spend_per_hh28_ly'] = r['macro_spend28_ly']/ly_hh if ly_hh else np.nan
    r['macro_ratio_ly'] = r['macro_spend28']/r['macro_spend28_ly'] if day>=392 else np.nan
    r['macro_growth'] = r['macro_spend28']/r['macro_spend_p28'] if r['macro_spend_p28'] else np.nan
    r['macro_ratio112_ly'] = r['macro_spend112']/r['macro_spend112_ly'] if day>=476 else np.nan
    wk = t.groupby('week_no')['sales_value'].sum()
    wcur = (day+8)//7
    l12 = wk.reindex(range(wcur-11, wcur+1)).values.astype(float)
    r['macro_wk_ratio_last3'] = np.nanmean(l12[-3:])/np.nanmean(l12[:-3]) if np.nanmean(l12[:-3]) else np.nan
    return pd.DataFrame({k: r[k] for k in r}, index=idx).astype(float)

mac = A.build_features(macro_fn)
print('macro table:', mac.shape, mac.columns.tolist())
ps = mac.drop_duplicates('snapshot_day').set_index('snapshot_day')
print(ps[['macro_spend28','macro_spend_per_hh28','macro_ratio_ly','macro_growth','macro_wk_ratio_last3']].round(3).to_string())
A.save_table(mac, 'macro.parquet')
print('saved macro.parquet; e014_dm.parquet already saved earlier')
