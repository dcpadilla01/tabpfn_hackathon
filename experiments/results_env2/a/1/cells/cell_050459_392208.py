import pandas as pd, numpy as np

allF = load_saved('allF.parquet')
print('ALL COLUMNS (%d):' % (len(allF.columns)))
print(list(allF.columns))

tt = train_targets().rename(columns={'future_spend_4w':'y'})
h = allF[allF.snapshot_day.isin([403,431])].merge(tt, on=['household_key','snapshot_day'])
print('\nholdout rows:', len(h), 'ymean %.1f' % h.y.mean())
FEATS = [c for c in allF.columns if c not in ('household_key','snapshot_day')]
corrs = {}
for c in FEATS:
    v = h[c].astype(float)
    if v.notna().sum() > 100:
        corrs[c] = np.corrcoef(v.fillna(v.mean()), h.y)[0,1]
top = pd.Series(corrs).abs().sort_values(ascending=False).head(15)
print('\ntop |corr| with y at 403/431:')
print(top.round(4))

# exact-match check: any column == y?
for c in FEATS:
    v = h[c]
    if v.notna().equals(h.y.notna()) and np.allclose(v.fillna(-1), h.y.fillna(-1)):
        print('EXACT MATCH WITH Y:', c)

# NaN pattern by snapshot day for the top-correlated column
c0 = top.index[0]
print('\nNaN rate of %s by snapshot_day:' % c0)
print(allF.groupby('snapshot_day')[c0].apply(lambda s: s.isna().mean()).round(3).to_string())
