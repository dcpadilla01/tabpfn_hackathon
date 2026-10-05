import numpy as np, pandas as pd

v = agent_api.snapshot(459)
tx = v.table('transactions')
raw = tx[['coupon_disc','coupon_match_disc','retail_disc']].fillna(0).sum(axis=1)
print('disc raw stats:', {k: round(float(x),3) for k,x in raw.describe().items()})
print('trans_time sample:', tx.trans_time.head(5).tolist(), 'dtype', tx.trans_time.dtype)
base = agent_api.load_saved('e013_stationary.parquet')
print('base', base.shape)
print('base@459 rows', base[base.snapshot_day==459].shape[0], '| households', len(v.households), '| type', type(v.households))

def fn(view):
    day = view.day
    tx = view.table('transactions')
    base = agent_api.load_saved('e013_stationary.parquet')
    b = base[base.snapshot_day == day].set_index('household_key')
    h = view.households
    if isinstance(h, pd.DataFrame):
        hh = pd.Index(h['household_key'] if 'household_key' in h.columns else h.index, name='household_key')
    elif isinstance(h, pd.Series):
        hh = pd.Index(h, name='household_key')
    else:
        hh = pd.Index(list(h), name='household_key')
    tx = tx[tx.household_key.isin(hh)].copy()
    d = tx[['coupon_disc','coupon_match_disc','retail_disc']].fillna(0).sum(axis=1)
    if float(d.median()) < 0: d = -d
    tx['d'] = d.clip(lower=0)
    F = {}; sp = {}
    for w in (28, 84, 364):
        s = tx[tx.day > day - w]
        F[f'disc_{w}'] = s.groupby('household_key')['d'].sum()
        sp[w] = s.groupby('household_key')['sales_value'].sum()
    F['disc_rate_28'] = F['disc_28'] / (sp[28] + F['disc_28'] + 1e-9)
    F['disc_rate_84'] = F['disc_84'] / (sp[84] + F['disc_84'] + 1e-9)
    l = tx[tx.day > day - 84]
    F['line_disc_share_84'] = l.assign(f=(l['d'] > 0).astype(float)).groupby('household_key')['f'].mean()
    trips = tx.drop_duplicates('basket_id')[['household_key','basket_id','day']]
    t84 = trips[trips.day > day - 84].copy()
    t84['dow'] = t84['day'] % 7
    def ent(x):
        c = np.bincount(x.to_numpy(), minlength=7)
        p = c[c > 0] / c.sum()
        return float(-(p * np.log(p)).sum())
    F['trip_dow_entropy_84'] = t84.groupby('household_key')['dow'].apply(ent)
    F['trip_dow_top_share_84'] = t84.groupby('household_key')['dow'].apply(
        lambda x: float(np.bincount(x.to_numpy(), minlength=7).max()) / len(x))
    trip_disc = tx.groupby('basket_id')['d'].max()
    t84['pdisc'] = t84['basket_id'].map(trip_disc > 0).astype(float)
    F['disc_trip_share_84'] = t84.groupby('household_key')['pdisc'].mean()
    tt = tx[tx.day > day - 84]
    F['morning_share_84'] = tt.assign(m=(tt.trans_time < 1200).astype(float)).groupby('household_key')['m'].mean()
    F['evening_share_84'] = tt.assign(e=(tt.trans_time >= 1700).astype(float)).groupby('household_key')['e'].mean()
    F['time_std_84'] = tt.groupby('household_key')['trans_time'].std()
    F['max_qty_84'] = tt.groupby('household_key')['quantity'].max()
    F['bulk_share_84'] = tt.assign(b=(tt.quantity >= 6).astype(float)).groupby('household_key')['b'].mean()
    nt28 = trips[trips.day > day - 28].groupby('household_key').size()
    u28 = tx[tx.day > day - 28].groupby('household_key')['quantity'].sum()
    F['units_per_trip_28'] = u28 / nt28
    X = pd.DataFrame(F)
    out = b.join(X, how='left')
    if 'snapshot_day' not in out.columns:
        out['snapshot_day'] = day
    out = out.reindex(hh)
    print(day, out.shape, 'newcols', len(F))
    return out

df = agent_api.build_features(fn)
print('FULL', df.shape)
path = agent_api.save_table(df, 'e019_rhythm')
print('saved:', path)


# ---- cell ----
import numpy as np, pandas as pd

def fn(view):
    day = view.day
    tx = view.table('transactions')
    base = agent_api.load_saved('e013_stationary.parquet')
    b = base[base.snapshot_day == day].set_index('household_key')
    hh = b.index
    tx = tx[tx.household_key.isin(hh)].copy()
    d = tx[['coupon_disc','coupon_match_disc','retail_disc']].fillna(0).sum(axis=1)
    if float(d.median()) < 0: d = -d
    tx['d'] = d.clip(lower=0)
    F = {}; sp = {}
    for w in (28, 84, 364):
        s = tx[tx.day > day - w]
        F[f'disc_{w}'] = s.groupby('household_key')['d'].sum()
        sp[w] = s.groupby('household_key')['sales_value'].sum()
    F['disc_rate_28'] = F['disc_28'] / (sp[28] + F['disc_28'] + 1e-9)
    F['disc_rate_84'] = F['disc_84'] / (sp[84] + F['disc_84'] + 1e-9)
    l = tx[tx.day > day - 84]
    F['line_disc_share_84'] = l.assign(f=(l['d'] > 0).astype(float)).groupby('household_key')['f'].mean()
    trips = tx.drop_duplicates('basket_id')[['household_key','basket_id','day']]
    t84 = trips[trips.day > day - 84].copy()
    t84['dow'] = t84['day'] % 7
    def ent(x):
        c = np.bincount(x.to_numpy(), minlength=7)
        p = c[c > 0] / c.sum()
        return float(-(p * np.log(p)).sum())
    F['trip_dow_entropy_84'] = t84.groupby('household_key')['dow'].apply(ent)
    F['trip_dow_top_share_84'] = t84.groupby('household_key')['dow'].apply(
        lambda x: float(np.bincount(x.to_numpy(), minlength=7).max()) / len(x))
    trip_disc = tx.groupby('basket_id')['d'].max()
    t84['pdisc'] = t84['basket_id'].map(trip_disc > 0).astype(float)
    F['disc_trip_share_84'] = t84.groupby('household_key')['pdisc'].mean()
    tt = tx[tx.day > day - 84]
    F['morning_share_84'] = tt.assign(m=(tt.trans_time < 1200).astype(float)).groupby('household_key')['m'].mean()
    F['evening_share_84'] = tt.assign(e=(tt.trans_time >= 1700).astype(float)).groupby('household_key')['e'].mean()
    F['time_std_84'] = tt.groupby('household_key')['trans_time'].std()
    F['max_qty_84'] = tt.groupby('household_key')['quantity'].max()
    F['bulk_share_84'] = tt.assign(b=(tt.quantity >= 6).astype(float)).groupby('household_key')['b'].mean()
    nt28 = trips[trips.day > day - 28].groupby('household_key').size()
    u28 = tx[tx.day > day - 28].groupby('household_key')['quantity'].sum()
    F['units_per_trip_28'] = u28 / nt28
    X = pd.DataFrame(F)
    out = b.join(X, how='left')
    if 'snapshot_day' not in out.columns:
        out['snapshot_day'] = day
    out = out.reindex(hh)
    print(day, out.shape, 'newcols', len(F))
    return out

df = agent_api.build_features(fn)
print('FULL', df.shape)
path = agent_api.save_table(df, 'e019_rhythm')
print('saved:', path)


# ---- cell ----
import numpy as np, pandas as pd

def fn(view, day):
    tx = view.table('transactions')
    base = agent_api.load_saved('e013_stationary.parquet')
    b = base[base.snapshot_day == day].set_index('household_key')
    hh = b.index
    tx = tx[tx.household_key.isin(hh)].copy()
    d = tx[['coupon_disc','coupon_match_disc','retail_disc']].fillna(0).sum(axis=1)
    if float(d.median()) < 0: d = -d
    tx['d'] = d.clip(lower=0)
    F = {}; sp = {}
    for w in (28, 84, 364):
        s = tx[tx.day > day - w]
        F[f'disc_{w}'] = s.groupby('household_key')['d'].sum()
        sp[w] = s.groupby('household_key')['sales_value'].sum()
    F['disc_rate_28'] = F['disc_28'] / (sp[28] + F['disc_28'] + 1e-9)
    F['disc_rate_84'] = F['disc_84'] / (sp[84] + F['disc_84'] + 1e-9)
    l = tx[tx.day > day - 84]
    F['line_disc_share_84'] = l.assign(f=(l['d'] > 0).astype(float)).groupby('household_key')['f'].mean()
    trips = tx.drop_duplicates('basket_id')[['household_key','basket_id','day']]
    t84 = trips[trips.day > day - 84].copy()
    t84['dow'] = t84['day'] % 7
    def ent(x):
        c = np.bincount(x.to_numpy(), minlength=7)
        p = c[c > 0] / c.sum()
        return float(-(p * np.log(p)).sum())
    F['trip_dow_entropy_84'] = t84.groupby('household_key')['dow'].apply(ent)
    F['trip_dow_top_share_84'] = t84.groupby('household_key')['dow'].apply(
        lambda x: float(np.bincount(x.to_numpy(), minlength=7).max()) / len(x))
    trip_disc = tx.groupby('basket_id')['d'].max()
    t84['pdisc'] = t84['basket_id'].map(trip_disc > 0).astype(float)
    F['disc_trip_share_84'] = t84.groupby('household_key')['pdisc'].mean()
    tt = tx[tx.day > day - 84]
    F['morning_share_84'] = tt.assign(m=(tt.trans_time < 1200).astype(float)).groupby('household_key')['m'].mean()
    F['evening_share_84'] = tt.assign(e=(tt.trans_time >= 1700).astype(float)).groupby('household_key')['e'].mean()
    F['time_std_84'] = tt.groupby('household_key')['trans_time'].std()
    F['max_qty_84'] = tt.groupby('household_key')['quantity'].max()
    F['bulk_share_84'] = tt.assign(b=(tt.quantity >= 6).astype(float)).groupby('household_key')['b'].mean()
    nt28 = trips[trips.day > day - 28].groupby('household_key').size()
    u28 = tx[tx.day > day - 28].groupby('household_key')['quantity'].sum()
    F['units_per_trip_28'] = u28 / nt28
    X = pd.DataFrame(F)
    out = b.join(X, how='left')
    if 'snapshot_day' not in out.columns:
        out['snapshot_day'] = day
    out = out.reindex(hh)
    print(day, out.shape, 'newcols', len(F))
    return out

df = agent_api.build_features(fn)
print('FULL', df.shape)
path = agent_api.save_table(df, 'e019_rhythm')
print('saved:', path)


# ---- cell ----
import numpy as np, pandas as pd

BASE = agent_api.load_saved('e013_stationary.parquet')
print('base', BASE.shape)

def fn(view, day):
    tx = view.table('transactions')
    b = BASE[BASE.snapshot_day == day].set_index('household_key')
    hh = b.index
    tx = tx[tx.household_key.isin(hh)].copy()
    d = tx[['coupon_disc','coupon_match_disc','retail_disc']].fillna(0).sum(axis=1)
    if float(d.median()) < 0: d = -d
    tx['d'] = d.clip(lower=0)
    F = {}; sp = {}
    for w in (28, 84, 364):
        s = tx[tx.day > day - w]
        F[f'disc_{w}'] = s.groupby('household_key')['d'].sum()
        sp[w] = s.groupby('household_key')['sales_value'].sum()
    F['disc_rate_28'] = F['disc_28'] / (sp[28] + F['disc_28'] + 1e-9)
    F['disc_rate_84'] = F['disc_84'] / (sp[84] + F['disc_84'] + 1e-9)
    l = tx[tx.day > day - 84]
    F['line_disc_share_84'] = l.assign(f=(l['d'] > 0).astype(float)).groupby('household_key')['f'].mean()
    trips = tx.drop_duplicates('basket_id')[['household_key','basket_id','day']]
    t84 = trips[trips.day > day - 84].copy()
    t84['dow'] = t84['day'] % 7
    def ent(x):
        c = np.bincount(x.to_numpy(), minlength=7)
        p = c[c > 0] / c.sum()
        return float(-(p * np.log(p)).sum())
    F['trip_dow_entropy_84'] = t84.groupby('household_key')['dow'].apply(ent)
    F['trip_dow_top_share_84'] = t84.groupby('household_key')['dow'].apply(
        lambda x: float(np.bincount(x.to_numpy(), minlength=7).max()) / len(x))
    trip_disc = tx.groupby('basket_id')['d'].max()
    t84['pdisc'] = t84['basket_id'].map(trip_disc > 0).astype(float)
    F['disc_trip_share_84'] = t84.groupby('household_key')['pdisc'].mean()
    tt = tx[tx.day > day - 84]
    F['morning_share_84'] = tt.assign(m=(tt.trans_time < 1200).astype(float)).groupby('household_key')['m'].mean()
    F['evening_share_84'] = tt.assign(e=(tt.trans_time >= 1700).astype(float)).groupby('household_key')['e'].mean()
    F['time_std_84'] = tt.groupby('household_key')['trans_time'].std()
    F['max_qty_84'] = tt.groupby('household_key')['quantity'].max()
    F['bulk_share_84'] = tt.assign(b=(tt.quantity >= 6).astype(float)).groupby('household_key')['b'].mean()
    nt28 = trips[trips.day > day - 28].groupby('household_key').size()
    u28 = tx[tx.day > day - 28].groupby('household_key')['quantity'].sum()
    F['units_per_trip_28'] = u28 / nt28
    X = pd.DataFrame(F)
    out = b.join(X, how='left')
    if 'snapshot_day' not in out.columns:
        out['snapshot_day'] = day
    out = out.reindex(hh)
    print(day, out.shape, 'newcols', len(F))
    return out

df = agent_api.build_features(fn)
print('FULL', df.shape)
path = agent_api.save_table(df, 'e019_rhythm')
print('saved:', path)


# ---- cell ----
import pandas as pd, numpy as np
df = agent_api.load_saved('e019_rhythm.parquet')
new = [c for c in df.columns if c not in ('household_key','snapshot_day')][-15:]
tr = df[df.snapshot_day <= 431]; va = df[df.snapshot_day >= 459]
print(f"{'col':22s} {'nan%':>6s} {'tr_mean':>9s} {'va_mean':>9s} {'tr_std':>8s} {'va_std':>8s}")
for c in new:
    print(f"{c:22s} {tr[c].isna().mean()*100:5.1f}% {tr[c].mean():9.3f} {va[c].mean():9.3f} {tr[c].std():8.3f} {va[c].std():8.3f}")
print('dup check:', df.duplicated(['household_key','snapshot_day']).sum(), 'rows', len(df))
