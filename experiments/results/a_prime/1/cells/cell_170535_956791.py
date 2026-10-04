
import agent_api as A
import pandas as pd, numpy as np

u = A.load_saved('e018_union_full.parquet')
tt = A.train_targets()
m = tt.merge(u, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w']

# 1) what are nratio_ly / nspend_ly / pct_* ? Correlate with spend_l1 and each other
cands = ['spend_l1','spend_l2','spend_l3','spend_l4','spend_l13','nspend28','nspend14',
         'nratio_ly','nspend_ly','pct_l1','pct_l4','pct_l13','peer_ratio','peer_ratio2',
         'cohort_prior4w','spend_ly4w','has_ly4w','ewm13_d','exp_trips28','basket_med_84',
         'momentum','trend_1v3','zero_frac_l13','act_w13','m_spend28_mean','m_spend28_med']
sub = m[cands + ['future_spend_4w']].copy()
corr = sub.corr(method='spearman')['future_spend_4w'].drop('future_spend_4w').sort_values(ascending=False)
print("Spearman corr with target (train rows):")
print(corr.round(3))
print()
print("corr nratio_ly vs spend_l1:", sub['nratio_ly'].corr(sub['spend_l1']).round(3),
      "| nspend_ly vs spend_l1:", sub['nspend_ly'].corr(sub['spend_l1']).round(3),
      "| pct_l1 vs spend_l1:", sub['pct_l1'].corr(sub['spend_l1']).round(3))
print("spend_ly4w NaN frac:", m['spend_ly4w'].isna().mean().round(3),
      "| spend_ly4w vs spend_l1 corr:", m['spend_ly4w'].corr(m['spend_l1']).round(3))
