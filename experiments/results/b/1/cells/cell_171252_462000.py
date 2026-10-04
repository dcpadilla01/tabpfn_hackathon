import agent_api as api
import pandas as pd, numpy as np

tt = api.train_targets()
e15 = api.load_saved('e015_base.parquet')
m = e15.merge(tt, on=['household_key','snapshot_day'], how='inner')

e11 = api.load_saved('e011_display.parquet')
e10 = api.load_saved('e010_composite.parquet')
bf  = api.baseline_features()

extra11 = [c for c in e11.columns if c not in e15.columns and c not in ('household_key','snapshot_day','spend_28')]
extra10 = [c for c in e10.columns if c not in e15.columns and c not in ('household_key','snapshot_day','spend_28')]
print('extra11:', extra11); print('extra10:', extra10)

m2 = m.merge(e11[['household_key','snapshot_day']+extra11], on=['household_key','snapshot_day'], how='left')
m2 = m2.merge(e10[['household_key','snapshot_day']+extra10], on=['household_key','snapshot_day'], how='left')
print('\n-- display/composite feature correlations with target and with e13:')
for c in extra11+extra10:
    x = m2[c]
    if pd.api.types.is_numeric_dtype(x) and x.notna().sum() > 100:
        print(f'{c:16s} corr_y={x.corr(m2["future_spend_4w"]):.4f}  corr_e13={x.corr(m2["e13"]):.3f}  na={x.isna().mean():.2f}')
    else:
        print(f'{c:16s} dtype={x.dtype} nunique={x.nunique()}')

mb = m.merge(bf[['household_key','snapshot_day','classification_2','snapshot_day_index','week_of_year']],
              on=['household_key','snapshot_day'], how='left')
print('\n-- baseline extras:')
print('classification_2 values:', mb['classification_2'].value_counts(dropna=False).to_dict())
for c in ['snapshot_day_index','week_of_year']:
    print(c, 'corr_y=', mb[c].corr(mb['future_spend_4w']).round(4))

# mean target by classification_2
print('\nmean target by classification_2:')
print(mb.groupby('classification_2')['future_spend_4w'].agg(['mean','count']).round(1))

# check 'streak' NaN corr (constant?) and missingness of top features
print('\nstreak nunique:', m['streak'].nunique(), 'na:', m['streak'].isna().mean())
top = ['e13','e6','b75','usual13','ew28','ew56','spend_84','weekly_rate_84','e13_4','usual13_4','mfly','w1','w13']
print('\nmissingness of top features:')
print(m[top].isna().mean().round(3))
print('\ndone')