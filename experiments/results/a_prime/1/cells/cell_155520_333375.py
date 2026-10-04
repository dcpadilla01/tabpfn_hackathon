api = agent_api
for name in ['e003_catmix.parquet','e001_txhist.parquet']:
    t = api.load_saved(name)
    print(name, t.shape)
    print(list(t.columns))
    print()
b = api.baseline_features()
print('baseline', b.shape)
print(list(b.columns))
tt = api.train_targets()
y = tt[api.TARGET]
print(y.describe())
print('zero share %.3f' % (y==0).mean())
print('quantiles', y.quantile([.5,.75,.9,.95,.99]).to_dict())