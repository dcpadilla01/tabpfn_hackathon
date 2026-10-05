import pandas as pd, numpy as np

names = ['e001_preds','e002_preds','e003_preds','e004_preds','e005_preds','e007_preds','e008_preds',
         'e009_preds','e010_preds','e011_preds','e012_preds','e014_preds','e016_preds','e017_preds',
         'e018_preds','e019_blend_preds']
tabs = {}
for n in names:
    try:
        df = load_saved(n + '.parquet')
        tabs[n] = df
    except Exception as e:
        print(n, 'ERR', type(e).__name__, str(e)[:80])

ref = tabs['e019_blend_preds']
print('ref rows', ref.shape, 'days', sorted(ref.snapshot_day.unique()))
print('dup keys:', ref.duplicated(['household_key','snapshot_day']).sum())

P = ref[['household_key','snapshot_day']].copy()
for n, df in tabs.items():
    t = df.rename(columns={'prediction': n})[['household_key','snapshot_day', n]]
    P = P.merge(t, on=['household_key','snapshot_day'], how='left')
cols = [n for n in tabs if n in P.columns]
print('merged', P.shape)
print('NaN counts per model:'); print(P[cols].isna().sum())

print('\n--- prediction distribution per model ---')
print(P[cols].describe().T[['mean','std','min','50%','max']].round(2))

print('\n--- correlation with e019 blend ---')
C = P[cols].corr()
print(C['e019_blend_preds'].round(4).sort_values())

print('\n--- mean abs deviation from e019 blend ---')
print(P[cols].sub(P['e019_blend_preds'], axis=0).abs().mean().round(2).sort_values())

print('\n--- train targets ---')
tt = train_targets()
print(tt.shape)
print(tt.groupby('snapshot_day').future_spend_4w.agg(['count','mean','median']).round(1))
print(tt.future_spend_4w.describe().round(2))

print('\n--- feature tables available ---')
for f in ['allF','e004_features','e002_features','e005_newfeats','camp_feats','lagfeats','lagfeats2','f_weekly']:
    try:
        d = load_saved(f + '.parquet')
        print(f, d.shape, 'cols:', list(d.columns)[:6], '... days:', sorted(d.snapshot_day.unique())[:20])
    except Exception as e:
        print(f, 'ERR', type(e).__name__, str(e)[:60])
