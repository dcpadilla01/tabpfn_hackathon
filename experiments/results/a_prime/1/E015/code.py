import agent_api, pandas as pd, numpy as np
pd.set_option('display.width', 250)
for name in ['e016_smoothed.parquet','e013_peers.parquet','micro.parquet','nf_candidates.parquet']:
    df = agent_api.load_saved(name)
    print('==', name, df.shape)
    print(list(df.columns))
    print()
t = agent_api.train_targets()
print('targets', t.shape)
print(t['future_spend_4w'].describe())
v = agent_api.snapshot()
tx = v.transactions
print('tx rows', len(tx), 'max day', tx['day'].max())
print(tx[['retail_disc','coupon_disc','coupon_match_disc','quantity','sales_value']].describe().round(2))
# population weekly spend per active household (seasonality check)
fd = tx.groupby('household_key')['day'].min()
tx2 = tx.copy(); tx2['wk'] = (tx2['day']+8)//7
wk_tot = tx2.groupby('wk')['sales_value'].sum()
base_n = pd.Series({w: (fd <= 7*w-8).sum() for w in wk_tot.index})
phh = (wk_tot/base_n)
print('weekly per-household spend, last 58 weeks:')
print(phh.tail(58).round(1).to_string())
print('overall mean', round(phh.mean(),1), 'cv', round(phh.std()/phh.mean(),3))

# ---- cell ----
import agent_api, pandas as pd, numpy as np
a = agent_api.load_saved('e012_dorm.parquet'); b = agent_api.load_saved('e016_smoothed.parquet')
ca, cb = set(a.columns)-{'household_key','snapshot_day'}, set(b.columns)-{'household_key','snapshot_day'}
print('e012 feats', len(ca), 'e016 feats', len(cb))
print('shared', len(ca&cb), 'only e012', sorted(ca-cb), 'only e016', len(cb-ca))
m = a.merge(b, on=['household_key','snapshot_day'], suffixes=('_a','_b'))
for c in sorted(ca&cb):
    x, y = m[c+'_a'], m[c+'_b']
    if x.dtype.kind in 'ifb' and y.dtype.kind in 'ifb':
        d = (x.fillna(-9e9)-y.fillna(-9e9)).abs().max()
        if d > 1e-9: print('DIFF', c, d)
print('merge shape', m.shape, '-> union feats', len(ca|cb))

# ---- cell ----
import agent_api, pandas as pd, numpy as np

def fn(view, snapshot_day):
    sd = snapshot_day
    tx = view.transactions
    tx84 = tx[tx['day'] > sd - 84]
    g = tx84.groupby('household_key')
    agg = g[['retail_disc','coupon_disc','coupon_match_disc']].sum()
    agg['disc_net_84'] = agg.sum(axis=1)          # net discount dollars (<=0)
    agg['coupon_disc_84'] = agg['coupon_disc']    # manufacturer coupon discounts
    agg['match_disc_84'] = agg['coupon_match_disc']
    agg = agg[['disc_net_84','coupon_disc_84','match_disc_84']]
    # coupon redemptions
    cr = view.coupon_redemptions
    cr84 = cr[cr['day'] > sd - 84]
    red = cr84.groupby('household_key').size().rename('coupon_redemptions_84')
    dsl = cr.groupby('household_key')['day'].max().rsub(sd).rename('dsl_coupon')
    out = agg.join(red, how='outer').join(dsl, how='outer')
    # cyclical seasonality from snapshot week
    wk = float(view.week)
    out['wk_sin'] = np.sin(2*np.pi*wk/52.0)
    out['wk_cos'] = np.cos(2*np.pi*wk/52.0)
    out['wk_sin2'] = np.sin(4*np.pi*wk/52.0)
    out['wk_cos2'] = np.cos(4*np.pi*wk/52.0)
    out = out.reindex(view.households)
    return out

X = agent_api.build_features(fn)
print(X.shape); print(X.dtypes); print(X.head())
base = agent_api.load_saved('e016_smoothed.parquet')
print('base', base.shape)
m = base.merge(X.reset_index(), on=['household_key','snapshot_day'], how='left')
print('merged', m.shape, 'new cols:', [c for c in m.columns if c not in base.columns])
print(m[['disc_net_84','coupon_disc_84','match_disc_84','coupon_redemptions_84','dsl_coupon','wk_sin','wk_cos']].describe().round(3))
path = agent_api.save_table(m, 'e017_disc_seasonal.parquet')
print(path)

# ---- cell ----
import agent_api, pandas as pd
base = agent_api.load_saved('e016_smoothed.parquet')
X = agent_api.load_saved('e017_disc_seasonal.parquet')
m = base.merge(X.drop(columns=['index']), on=['household_key','snapshot_day'], how='left')
print(m.shape, len(m)-len(base))
path = agent_api.save_table(m, 'e017_disc_seasonal.parquet')
print(path)

# ---- cell ----
import agent_api, pandas as pd
base = agent_api.load_saved('e016_smoothed.parquet')
big = agent_api.load_saved('e017_disc_seasonal.parquet')
newcols = ['disc_net_84','coupon_disc_84','match_disc_84','coupon_redemptions_84','dsl_coupon','wk_sin','wk_cos','wk_sin2','wk_cos2']
X = big[['household_key','snapshot_day']+newcols]
print(X.shape)
m = base.merge(X, on=['household_key','snapshot_day'], how='left')
print('merged', m.shape, 'dup rows:', len(m)-len(base))
assert len(m)==len(base) and m.shape[1]==127+9
path = agent_api.save_table(m, 'e017_v2.parquet')
print(path)