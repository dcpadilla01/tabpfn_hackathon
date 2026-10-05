import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
t = A.load_saved('e011_discounts.parquet')
tt = A.train_targets()
m = tt.merge(t, on=['household_key','snapshot_day'], how='inner')
y = m['future_spend_4w'].astype(float).values
num_cols = [c for c in t.columns if c not in ('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(m[c])]
rows=[]
for c in num_cols:
    v = m[c].astype(float); vn = v.fillna(v.median()).values
    if vn.std()==0: continue
    rows.append((c, float(np.corrcoef(vn,y)[0,1]), float(np.abs(vn-y).mean())))
d = pd.DataFrame(rows, columns=['feat','corr','naiveMAE']).sort_values('corr', key=lambda s: s.abs(), ascending=False)
print(d.round(3).to_string())
# zero vs nonzero structure
z = y==0
print('\nzero rows:', z.sum())
for c in ['spend_28','spend_84','ew_28','ew_84','days_since_last','active_28','spend_56']:
    v = m[c].astype(float).fillna(m[c].median()).values
    print(f'{c:16s} mean|z={v[z].mean():8.2f} |nz={v[~z].mean():8.2f} | corr_nonz={np.corrcoef(v[~z],y[~z])[0,1]:.3f}')