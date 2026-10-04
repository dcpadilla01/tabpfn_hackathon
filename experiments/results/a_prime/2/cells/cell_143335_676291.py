import pandas as pd
import agent_api

def dbg(view, snapshot_day):
    hh = view.households
    print('day', snapshot_day, 'type', type(hh))
    if hh is not None:
        print('len', len(hh))
    # return minimal frame indexed by household_key
    if isinstance(hh, pd.DataFrame):
        idx = pd.Index(hh['household_key'].values, name='household_key')
    else:
        idx = pd.Index(list(hh), name='household_key')
    return pd.DataFrame({'x': 1.0}, index=idx)

out = agent_api.build_features(dbg)
print('OK', out.shape)
print(out.head())
