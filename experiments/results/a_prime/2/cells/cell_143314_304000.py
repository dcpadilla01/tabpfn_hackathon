import pandas as pd, numpy as np
import agent_api
v = agent_api.snapshot(95)
hh = v.households
print(type(hh))
print(hh if isinstance(hh, pd.Series) else hh.head())
print('len', len(hh), 'unique', pd.unique(pd.Series(list(hh))).size if not hasattr(hh,'columns') else pd.unique(hh['household_key']).size)
base = agent_api.baseline_features()
print('base shape', base.shape, 'cols', list(base.columns)[:12])
print('base index name:', base.index.name, 'dup idx:', base.index.duplicated().sum() if base.index.name else 'n/a')
