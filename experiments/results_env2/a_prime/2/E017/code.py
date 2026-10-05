
import agent_api as A
import pandas as pd, numpy as np

t = A.load_saved('e016_churn_gapratio.parquet')
print('shape', t.shape)
for i, c in enumerate(t.columns):
    print(i, c, t[c].dtype)

tt = A.train_targets()
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','std','count']))
print(A.snapshot_days())


# ---- cell ----

import agent_api as A
v = A.snapshot()
p = v.products
print(p.department.value_counts().head(30))
tr = v.transactions
print(tr.shape)
print(tr.head(3))
# check a household's history for gap/window conventions
h = A.history(v.households.iloc[0], as_of_day=459)
print(h[['day','basket_id','sales_value']].tail(5))


# ---- cell ----

import agent_api as A
import pandas as pd, numpy as np

t = A.load_saved('e016_churn_gapratio.parquet')
g = t.groupby('snapshot_day')

new = pd.DataFrame(index=t.index)
new['pctl_s28']    = g['spend_28'].rank(pct=True)
new['pctl_ew28']   = g['ew_28'].rank(pct=True)
new['pctl_s84']    = g['spend_84'].rank(pct=True)
new['pctl_dsl']    = g['days_since_last'].rank(pct=True)
new['rel_s28_med']  = t['spend_28'] / t['snap_med_s28']
new['rel_ew28_med'] = t['ew_28'] / t['snap_med_ew28']
new['rel_s84_med']  = t['spend_84'] / t['snap_med_s84']
new['rel_dsl_med']  = t['dsl'] / t['snap_mean_dsl']
new['s28_minus_med'] = t['spend_28'] - t['snap_med_s28']
new['ew28_over_p90'] = t['ew_28'] / t['snap_p90_s28']

out = pd.concat([t, new], axis=1)
print(out.shape, out[['snapshot_day']].nunique().values)
print(new.describe().T[['mean','std','min','max']])
path = A.save_table(out, 'e017_xsec_rank.parquet')
print(path)


# ---- cell ----

import agent_api as A
import pandas as pd, numpy as np

t = A.load_saved('e016_churn_gapratio.parquet')
g = t.groupby('snapshot_day')

new = pd.DataFrame(index=t.index)
new['pctl_s28']     = g['spend_28'].rank(pct=True)
new['pctl_ew28']    = g['ew_28'].rank(pct=True)
new['pctl_s84']     = g['spend_84'].rank(pct=True)
new['pctl_dsl']     = g['days_since_last'].rank(pct=True)
new['rel_s28_med']  = t['spend_28'] / t['snap_med_s28']
new['rel_ew28_med'] = t['ew_28'] / g['ew_28'].transform('median')
new['rel_s84_med']  = t['spend_84'] / t['snap_med_s84']
new['rel_dsl_med']  = t['dsl'] / t['snap_mean_dsl']
new['s28_minus_med'] = t['spend_28'] - t['snap_med_s28']
new['ew28_over_p90'] = t['ew_28'] / t['snap_p90_s28']

out = pd.concat([t, new], axis=1)
print(out.shape)
print(new.describe().T[['mean','std','min','max']])
path = A.save_table(out, 'e017_xsec_rank.parquet')
print(path)


# ---- cell ----

import agent_api as A
import pandas as pd, numpy as np

t = A.load_saved('e016_churn_gapratio.parquet')
g = t.groupby('snapshot_day')

med_s28  = g['spend_28'].transform('median')
med_ew28 = g['ew_28'].transform('median')
med_s84  = g['spend_84'].transform('median')
mean_dsl = g['dsl'].transform('mean')
p90_s28  = g['spend_28'].transform(lambda s: s.quantile(0.9))

new = pd.DataFrame(index=t.index)
new['pctl_s28']      = g['spend_28'].rank(pct=True)
new['pctl_ew28']     = g['ew_28'].rank(pct=True)
new['pctl_s84']      = g['spend_84'].rank(pct=True)
new['pctl_dsl']      = g['dsl'].rank(pct=True)
new['rel_s28_med']   = t['spend_28'] / med_s28
new['rel_ew28_med']  = t['ew_28'] / med_ew28
new['rel_s84_med']   = t['spend_84'] / med_s84
new['rel_dsl_med']   = t['dsl'] / mean_dsl
new['s28_minus_med'] = t['spend_28'] - med_s28
new['ew28_over_p90'] = t['ew_28'] / p90_s28

out = pd.concat([t, new], axis=1)
print(out.shape)
print(new.describe().T[['mean','std','min','max']])
path = A.save_table(out, 'e017_xsec_rank.parquet')
print(path)
