def probe(view, day):
    import numpy as np
    hh = view.households
    print('type', type(hh), 'n', len(hh), 'first', hh[:5] if hasattr(hh,'__getitem__') else None)
    return pd.DataFrame({'x': np.ones(len(hh))}, index=pd.Index(hh, name='household_key'))

df = agent_api.build_features(probe)
print(df.shape, df.snapshot_day.unique())
print(df.head())
