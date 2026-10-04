import agent_api as A
import pandas as pd, numpy as np

e3 = A.load_saved('e003_catmix.parquet')
e3 = e3.copy()
fake = e3.tenure < 364
print('rows with fake l13:', int(fake.sum()), 'of', len(e3))
e3['spend_l13'] = e3.spend_l13.where(~fake, np.nan)
e3['has_real_l13'] = (~fake).astype(float)
# also a ratio of seasonal lag to recent mean (only meaningful when real)
e3['l13_over_recent'] = e3.spend_l13 / (e3.spend_l123_mean + 1e-6)
print(e3[['spend_l13','has_real_l13','l13_over_recent']].describe())
A.save_table(e3, 'e010_l13fix.parquet')
print('saved', e3.shape)
