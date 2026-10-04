import warnings; warnings.filterwarnings('ignore')
t = load_saved('e007_lagseq.parquet')
m = train_targets().merge(t, on=['household_key','snapshot_day'], how='left')
num = m.select_dtypes(include=[np.number])
num = num.loc[:, num.std() > 0]  # drop zero-variance
cor = num.corrwith(m.future_spend_4w).drop('future_spend_4w', errors='ignore').sort_values()
print('TOP+ :'); print(cor.tail(18).round(3))
print('TOP- :'); print(cor.head(10).round(3))
# check lag_spend_13 vs own_ly_spend4w
print('corr lag13 vs own_ly:', m[['lag_spend_13','own_ly_spend4w']].corr().iloc[0,1].round(3))
print('corr lag1 vs spend28:', m[['lag_spend_1','spend28']].corr().iloc[0,1].round(3))
# tenure dist by snapshot
print(m.groupby('snapshot_day').tenure.mean().round(0))
print(m.groupby('snapshot_day').future_spend_4w.mean().round(1))
p = snapshot().products
print('products', p.shape, p.columns.tolist())
print('n commodities', p.commodity_desc.nunique(), 'n depts', p.department.nunique())
print(p.brand.value_counts().head())
