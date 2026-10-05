
import pandas as pd, numpy as np
import agent_api as A

print(A.snapshot_days())
s = A.snapshot()
print('view:', type(s), 'day', s.day, 'week', s.week)
tx = s.transactions
print('tx shape', tx.shape)
print(tx.head(3))
print(tx[['sales_value','quantity','coupon_disc','retail_disc','coupon_match_disc']].describe())
print('households in tx:', tx.household_key.nunique())
hh = s.households
print('households type', type(hh), 'len', len(hh))
print(list(hh[:5]) if not isinstance(hh, pd.DataFrame) else hh.head())
h = A.history(tx.household_key.iloc[0])
print('history shape', h.shape)
print(h.head(3))
d = s.demographics
print('demographics', d.shape)
ct = s.campaign_targets
print('campaign_targets', ct.shape)
print(ct.description.value_counts().head())
cr = s.coupon_redemptions
print('coupon_redemptions', cr.shape)

# quick predictive check: past 28d spend vs target at snapshot 431
tt = A.train_targets()
print('targets', tt.shape)
print(tt.future_spend_4w.describe())
s431 = A.snapshot(431)
t431 = s431.transactions
sub = tt[tt.snapshot_day == 431].set_index('household_key')
t = t431[t431.day > 431-28]
sp28 = t.groupby('household_key').sales_value.sum().reindex(sub.index).fillna(0)
t84 = t431[t431.day > 431-84]
sp84 = t84.groupby('household_key').sales_value.sum().reindex(sub.index).fillna(0)
print('corr sp28 vs target', np.corrcoef(sp28, sub.future_spend_4w)[0,1])
print('corr sp84 vs target', np.corrcoef(sp84, sub.future_spend_4w)[0,1])
print('MAE of predicting target=sp28:', (sp28 - sub.future_spend_4w).abs().mean())
print('MAE of predicting target=sp84/3:', (sp84/3 - sub.future_spend_4w).abs().mean())
print('MAE of predicting target=mean:', (sub.future_spend_4w.mean() - sub.future_spend_4w).abs().mean())


# ---- cell ----

import pandas as pd, numpy as np
import agent_api as A

tt = A.train_targets()
print('targets', tt.shape)
print(tt.future_spend_4w.describe())
print(tt.groupby('snapshot_day').size())

# households per snapshot: first purchase >= 84 days earlier
s = A.snapshot()
tx = s.transactions
first = tx.groupby('household_key').day.min()
print('first day quantiles:', first.quantile([0,.25,.5,.75,1]).values)

# quick predictive check at snapshot 431
sub = tt[tt.snapshot_day == 431].set_index('household_key')
t431 = tx[tx.day <= 431]
sp28 = t431[t431.day > 431-28].groupby('household_key').sales_value.sum().reindex(sub.index).fillna(0)
sp84 = t431[t431.day > 431-84].groupby('household_key').sales_value.sum().reindex(sub.index).fillna(0)
sp364 = t431[t431.day > 431-364].groupby('household_key').sales_value.sum().reindex(sub.index).fillna(0)
y = sub.future_spend_4w
print('corr sp28', np.corrcoef(sp28, y)[0,1], 'corr sp84', np.corrcoef(sp84, y)[0,1], 'corr sp364', np.corrcoef(sp364, y)[0,1])
print('MAE 0:', y.abs().mean(), 'MAE mean:', (y.mean()-y).abs().mean())
print('MAE sp28:', (sp28-y).abs().mean(), 'MAE sp84/3:', (sp84/3-y).abs().mean(), 'MAE sp364/13:', (sp364/13-y).abs().mean())
# blend
for w in [0.3,0.5,0.7]:
    print(f'MAE blend {w}*sp28+(1-w)*sp364/13:', (w*sp28+(1-w)*sp364/13-y).abs().mean())
print('zero fraction', (y==0).mean())


# ---- cell ----

import pandas as pd, numpy as np, time
import agent_api as A

def feat(view, sd):
    hh = view.households
    if callable(hh): hh = hh()
    hh = pd.Index(np.asarray(hh).ravel(), name='household_key')
    tx = view.table('transactions')
    tx = tx[tx.household_key.isin(set(hh))]
    out = pd.DataFrame(index=hh)
    base = tx.groupby('household_key').day
    out['tenure'] = (sd - base.min()).reindex(hh)
    out['days_since_last'] = (sd - base.max()).reindex(hh)

    bs = tx.groupby(['household_key','basket_id']).agg(sp=('sales_value','sum'), d0=('day','min')).reset_index()

    for w in [28,56,84,168,364,728]:
        t = tx[tx.day > sd-w]
        g = t.groupby('household_key')
        a = g.agg(sp=('sales_value','sum'), trips=('basket_id','nunique'),
                  prods=('product_id','nunique'), stores=('store_id','nunique'), qty=('quantity','sum'))
        for c in a.columns: out[f'{c}{w}'] = a[c].reindex(hh).fillna(0)
        bw = bs[bs.d0 > sd-w]
        gb = bw.groupby('household_key').sp
        out[f'avgbs{w}'] = gb.mean().reindex(hh)
        out[f'maxbs{w}'] = gb.max().reindex(hh)
        out[f'nact{w}'] = bw.groupby('household_key').d0.nunique().reindex(hh).fillna(0)

    for w in [84,364]:
        t = tx[tx.day > sd-w]
        g = t.groupby('household_key')
        out[f'cdisc{w}'] = g.coupon_disc.sum().reindex(hh).fillna(0)
        out[f'rdisc{w}'] = g.retail_disc.sum().reindex(hh).fillna(0)

    out['trend_28_56'] = (out.sp28+1)/(out.sp56+1)
    out['trend_84'] = (out.sp28+1)/(out.sp84/3+1)
    out['sp28_rate'] = out.sp28/28
    out['sp84_rate'] = out.sp84/84
    out['sp364_rate'] = out.sp364/364
    t = tx[(tx.day > sd-392) & (tx.day <= sd-364)]
    out['sp_lag1y'] = t.groupby('household_key').sales_value.sum().reindex(hh).fillna(0)

    d = view.demographics.set_index('household_key').reindex(hh)
    for c in d.columns:
        out['d_'+c] = d[c]
    out['has_demo'] = d['classification_1'].notna().astype(int)
    out['snapshot_day'] = sd
    out['week_mod52'] = ((sd+8)//7) % 52
    if sd == 95:
        print('shape', out.shape, 'nan frac', out.isna().mean().mean())
    return out

t0=time.time()
df = A.build_features(feat)
print('build time', time.time()-t0, df.shape)
print(df.head(3).T.head(30))
p = A.save_table(df, 'e001_history')
print(p)
