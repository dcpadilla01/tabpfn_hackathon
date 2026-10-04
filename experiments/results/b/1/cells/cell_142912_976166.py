def probe(view, day):
    print('inside build_features: day', day, 'households type', type(view.households))
    import numpy as np
    hh = view.households
    print('n hh', len(hh) if hh is not None else None)
    return pd.DataFrame({'x': [1.0]}, index=pd.Index(hh[:3] if hh is not None else [], name='household_key'))

df = agent_api.build_features(probe)
print(df.head())
