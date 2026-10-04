
import agent_api as A
import pandas as pd, numpy as np

u = A.load_saved('e018_union_full.parquet')
tt = A.train_targets()
m = tt.merge(u, on=['household_key','snapshot_day'], how='left')

ten = m[m.tenure_x >= 364]
print("tenured rows:", len(ten))
for c in ['spend_l13','spend_ly4w','pct_l13','nspend_ly','nratio_ly']:
    print(f"{c:12s} corr(target): {ten[c].corr(ten['future_spend_4w']):.3f}  corr(spend_l1): {ten[c].corr(ten['spend_l1']):.3f}  mean {ten[c].mean():.1f}  zero-frac {(ten[c]==0).mean():.2f}")

# distribution of spend_l13 vs spend_l1 for tenured
print("\nspend_l13 describe (tenured):"); print(ten['spend_l13'].describe().round(1))
print("spend_l1  describe (tenured):"); print(ten['spend_l1'].describe().round(1))
# check pct_l13 definition: corr with spend_l13 and spend_l1
print("\npct_l13 vs spend_l13:", ten['pct_l13'].corr(ten['spend_l13']).round(3), " vs spend_l1:", ten['pct_l13'].corr(ten['spend_l1']).round(3))
# has_real_l13 flag coverage
print("has_real_l13 mean (all):", m['has_real_l13'].mean().round(3), " (tenured):", ten['has_real_l13'].mean().round(3))
