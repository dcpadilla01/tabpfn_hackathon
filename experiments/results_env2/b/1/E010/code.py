
df8 = load_saved('e008_level_shape.parquet')
print('E008 table:', df8.shape)
print(sorted([c for c in df8.columns if c not in ('household_key','snapshot_day')]))

e1 = load_saved('e001_history.parquet'); e3 = load_saved('e003_full.parquet')
e6 = load_saved('e006_cadence.parquet'); e7 = load_saved('e007_temporal.parquet')
df2 = load_saved('e002_mix.parquet')
print('\nE002 mix-only cols:', sorted(set(df2.columns)-set(e1.columns)-{'household_key','snapshot_day'}))
print('\nE006 cadence-only cols:', sorted(set(e6.columns)-set(e3.columns)-{'household_key','snapshot_day'}))
print('\nE007 temporal-only cols:', sorted(set(e7.columns)-set(e3.columns)-{'household_key','snapshot_day'}))

v = snapshot()
t = v.transactions
print('\ndiscount signs:')
print(t[['sales_value','quantity','retail_disc','coupon_disc','coupon_match_disc']].describe().loc[['mean','min','max']])


# ---- cell ----

e8 = load_saved('e008_level_shape.parquet')
print('e8', e8.shape)

def gross_feats(view, sd):
    t = view.table('transactions')
    t = t[t.day <= sd].copy()
    t['gross'] = (t.sales_value - t.retail_disc - t.coupon_disc - t.coupon_match_disc).clip(lower=0)
    out = {}
    for name, lo in [('28', sd-27), ('56', sd-55), ('84', sd-83), ('168', sd-167), ('364', sd-363)]:
        w = t[(t.day > lo) & (t.day <= sd)]
        g = w.groupby('household_key')['gross'].sum()
        n = w.groupby('household_key')['sales_value'].sum()
        out['gsp'+name] = np.log1p(g)
        if name in ('28','84','364'):
            out['gnet'+name] = (n/g).where(g > 0).clip(0, 1.5)
    # EWMA of gross spend, half-lives 28/56 days
    tt = t[t.day > sd-364]
    for hl, nm in [(28,'gewma28'), (56,'gewma56')]:
        w = np.power(0.5, (sd - tt.day) / hl)
        out[nm] = np.log1p((w * tt.gross).groupby(tt.household_key).sum())
    # gross momentum (log diff 28 vs 84) and seasonal gross (same 4w last year)
    out['r_gsp28_84'] = out['gsp28'] - out['gsp84']
    w = t[(t.day > sd-391) & (t.day <= sd-363)]
    out['gsp_seas1y'] = np.log1p(w.groupby('household_key')['gross'].sum())
    out['dgn364'] = out['gsp364'] - np.log1p(t[t.day > sd-364].groupby('household_key')['sales_value'].sum())
    df = pd.DataFrame(out)
    df = df.fillna({c: 0.0 for c in df.columns if c.startswith(('gsp','gewma'))})
    return df

nf = build_features(gross_feats)
print('new feats', nf.shape, list(nf.columns))
print(nf.isna().sum())

df = e8.merge(nf.reset_index(), on=['household_key','snapshot_day'], how='left')
print('merged', df.shape, 'NaN rows in new cols:', df[list(nf.columns)].isna().any(axis=1).sum())
assert df.shape[0] == e8.shape[0]
p = save_table(df, 'e010_gross.parquet')
print(p)
