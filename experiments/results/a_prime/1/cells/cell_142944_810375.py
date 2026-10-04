import agent_api as api
import pandas as pd

def fn(view, snapshot_day):
    print('day', snapshot_day, 'households type:', type(view.households), 'week:', view.week)
    tx = view.transactions
    print('tx shape', tx.shape, 'max day', tx.day.max())
    return pd.DataFrame({'x': 1.0}, index=pd.Index(tx.household_key.unique(), name='household_key'))

full = api.build_features(fn)
print(full.shape)
print(full.head())
