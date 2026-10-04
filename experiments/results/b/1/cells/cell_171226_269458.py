import agent_api as api
import pandas as pd, numpy as np

bf = api.baseline_features()
print('baseline cols:', list(bf.columns))

tt = api.train_targets()
print('train_targets:', tt.shape, list(tt.columns))

e15 = api.load_saved('e015_base.parquet')
m = e15.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged:', m.shape)

num = [c for c in e15.columns if c not in ('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(m[c])]
corrs = {}
for c in num:
    x = m[c]
    if x.notna().sum() > 100:
        corrs[c] = x.corr(m['future_spend_4w'])
cs = pd.Series(corrs).sort_values()
print('\n--- E015 weakest |corr| (bottom 25):')
print(cs.abs().sort_values().head(25).round(4))
print('\n--- E015 strongest |corr| (top 20):')
print(cs.abs().sort_values().tail(20).round(4))

# missing families: display/mailer (e011), composites (e010)
e11 = api.load_saved('e011_display.parquet')
e10 = api.load_saved('e010_composite.parquet')
extra11 = [c for c in e11.columns if c not in e15.columns and c not in ('household_key','snapshot_day')]
extra10 = [c for c in e10.columns if c not in e15.columns and c not in ('household_key','snapshot_day')]
print('\nextra11:', extra11)
print('extra10:', extra10)
m2 = m.merge(e11[['household_key','snapshot_day']+extra11], on=['household_key','snapshot_day'], how='left')
m2 = m2.merge(e10[['household_key','snapshot_day']+extra10], on=['household_key','snapshot_day'], how='left')
for c in extra11+extra10:
    x = m2[c]
    if x.notna().sum() > 100 and pd.api.types.is_numeric_dtype(x):
        print(f'{c:18s} corr={x.corr(m2["future_spend_4w"]):.4f}  corr_with_e13={x.corr(m2["e13"]):.3f}')

# baseline calendar/demographic corr
mb = m.merge(bf.drop(columns=[c for c in bf.columns if c in e15.columns]), on=['household_key','snapshot_day'], how='left')
for c in ['snapshot_day_index','week_of_year','classification_2']:
    if c in mb.columns:
        x = mb[c]
        print(f'{c:20s} dtype={x.dtype} nunique={x.nunique()} corr={pd.to_numeric(x, errors="coerce").corr(mb["future_spend_4w"]) if pd.api.types.is_numeric_dtype(x) else "cat"}')
print('\ncalls ok')