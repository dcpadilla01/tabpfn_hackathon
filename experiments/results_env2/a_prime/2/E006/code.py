import agent_api, pandas as pd, numpy as np

e5 = agent_api.load_saved('e005_decay_gapcv.parquet')
print('E005 cols:', list(e5.columns))
print('E005 shape:', e5.shape)
t = agent_api.train_targets()
m = e5.merge(t, on=['household_key','snapshot_day'])
y = m['future_spend_4w']
print('target describe:', y.describe().round(2).to_dict())
print('zero share: %.3f' % (y == 0).mean())
c = m.corr(numeric_only=True)['future_spend_4w'].drop('future_spend_4w')
print('E005 feature corr with target (by |corr|):')
print(c.loc[c.abs().sort_values(ascending=False).index].round(3).to_string())

def fn(view, d):
    tx = view.table('transactions')
    hh = pd.Index(view.households)
    tx = tx[tx['household_key'].isin(hh)]
    days = tx['day']
    def wsum(lo, hi):
        msk = (days >= lo) & (days <= hi)
        return tx.loc[msk].groupby('household_key')['sales_value'].sum()
    def wtrips(lo, hi):
        msk = (days >= lo) & (days <= hi)
        return tx.loc[msk].groupby('household_key')['basket_id'].nunique()
    s28 = wsum(d-27, d).reindex(hh).fillna(0.0)
    sp28 = wsum(d-55, d-28).reindex(hh).fillna(0.0)
    s365 = wsum(d-364, d).reindex(hh).fillna(0.0)
    out = pd.DataFrame(index=hh)
    out['spend_prev28'] = sp28
    out['mom28_delta'] = s28 - sp28
    out['mom28_ratio'] = (s28 + 1.0) / (sp28 + 1.0)
    out['accel'] = s28/28.0 - s365/365.0
    rate13 = s365/13.0
    out['recent_vs_typ'] = (s28 + 1.0)/(rate13 + 1.0)
    def weekly(ndays):
        r = tx[days > d - ndays]
        wk = (d - r['day'])//7
        p = r.assign(_w=wk).groupby(['household_key','_w'])['sales_value'].sum().unstack(fill_value=0.0)
        p = p.reindex(columns=range(ndays//7), fill_value=0.0).reindex(hh, fill_value=0.0)
        return p
    p12 = weekly(84)
    x = np.arange(12.0); xc = x - x.mean()
    out['slope_12w'] = p12.values @ xc / (xc @ xc)
    a = weekly(182).values
    out['wcv_26'] = a.std(axis=1)/(a.mean(axis=1) + 1.0)
    out['lag364_4w'] = wsum(d-363, d-336).reindex(hh).fillna(0.0)
    out['lag336_4w'] = wsum(d-335, d-308).reindex(hh).fillna(0.0)
    out['lag364_ratio'] = (out['lag364_4w'] + 1.0)/(rate13 + 1.0)
    tr28 = wtrips(d-27, d).reindex(hh).fillna(0.0)
    trp = wtrips(d-55, d-28).reindex(hh).fillna(0.0)
    out['trips_ratio'] = (tr28 + 0.5)/(trp + 0.5)
    b28 = (s28/tr28.replace(0, np.nan)).fillna(0.0)
    bp = (sp28/trp.replace(0, np.nan)).fillna(0.0)
    out['basket_ratio'] = (b28 + 1.0)/(bp + 1.0)
    last = tx.groupby('household_key')['day'].max().reindex(hh)
    out['days_since_trip'] = (d - last).fillna(999.0)
    out['active_days_28'] = tx[days > d-28].groupby('household_key')['day'].nunique().reindex(hh).fillna(0.0)
    age = (d - days).astype(float)
    ew28 = (tx['sales_value']*np.power(0.5, age/28.0)).groupby(tx['household_key']).sum().reindex(hh).fillna(0.0)
    ew180 = (tx['sales_value']*np.power(0.5, age/180.0)).groupby(tx['household_key']).sum().reindex(hh).fillna(0.0)
    out['ew_mom'] = (ew28 + 1.0)/(ew180 + 1.0)
    out['tenure'] = (d - tx.groupby('household_key')['day'].min().reindex(hh)).astype(float)
    return out

df = agent_api.build_features(fn)
if 'household_key' not in df.columns:
    df = df.reset_index()
newcols = [c for c in df.columns if c not in e5.columns and c not in ('household_key','snapshot_day')]
print('new features:', newcols)
print('built shape:', df.shape)
full = e5.merge(df[['household_key','snapshot_day'] + newcols], on=['household_key','snapshot_day'], how='inner')
print('merged shape:', full.shape)
path = agent_api.save_table(full, 'e006_momentum_seasonal.parquet')
print('saved:', path)
mt = full.merge(t, on=['household_key','snapshot_day'])
cn = mt[newcols + ['future_spend_4w']].corr()['future_spend_4w'].drop('future_spend_4w')
print('new feature corr with target (by |corr|):')
print(cn.loc[cn.abs().sort_values(ascending=False).index].round(3).to_string())
