
t6 = agent_api.load_saved('e006_zero_inflation.parquet')
t7 = agent_api.load_saved('e007_ar_lags.parquet')
print('e006', t6.shape); print(list(t6.columns))
print('e007', t7.shape); print([c for c in t7.columns if c not in t6.columns])
tt = agent_api.train_targets()
print('targets', tt.shape)
print(tt.future_spend_4w.describe())
print('zero share train:', (tt.future_spend_4w==0).mean())
