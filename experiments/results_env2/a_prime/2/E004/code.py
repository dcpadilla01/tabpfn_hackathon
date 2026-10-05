import pandas as pd, numpy as np
e1 = agent_api.load_saved('e001_recent_behavior.parquet')
print("E001 cols:", list(e1.columns))
print("shape:", e1.shape)
tt = agent_api.train_targets()
print("targets shape:", tt.shape)
print(tt['future_spend_4w'].describe())
print("zero share:", (tt['future_spend_4w']==0).mean())
m = e1.merge(tt, on=['household_key','snapshot_day'], how='inner')
print("merged:", m.shape)
num = [c for c in e1.columns if c not in ('household_key','snapshot_day')]
cor = m[num+['future_spend_4w']].corr(numeric_only=True)['future_spend_4w'].drop('future_spend_4w')
print(cor.reindex(cor.abs().sort_values(ascending=False).index).round(3))
v = agent_api.snapshot(459)
tx = v.table('transactions')
print("tx rows@459:", len(tx), "hh needing rows:", len(v.households))


# ---- cell ----
import pandas as pd, numpy as np
v = agent_api.snapshot()
tx = v.table('transactions')
print("tx rows@459:", len(tx), "max day:", tx.day.max(), "n hh:", tx.household_key.nunique())
# seasonality: total spend by 4-week block
tx['blk'] = ((tx.day-1)//28)
s = tx.groupby('blk').sales_value.sum()
print(s.round(0).to_string())


# ---- cell ----
import pandas as pd, numpy as np

def make(view, day):
    hh = pd.Index(view.households, name='household_key')
    tx = view.table('transactions')
    tx = tx[tx.household_key.isin(hh)]
    f = pd.DataFrame(index=hh)
    age = (day - tx.day).astype(float)
    # recency-weighted spend
    for hl in (7, 14, 28, 56):
        w = np.power(0.5, age / hl)
        f[f'ew_{hl}'] = (tx.sales_value * w).groupby(tx.household_key).sum().reindex(hh).fillna(0.0)
    # very recent windows
    for win in (7, 14):
        f[f'spend_{win}'] = tx[tx.day > day - win].groupby('household_key').sales_value.sum().reindex(hh).fillna(0.0)
    # same relative 4-week window one year (and +/-4wk) earlier
    for lag in (336, 364, 392):
        m = tx[(tx.day > day - lag) & (tx.day <= day - lag + 28)]
        f[f'spend_lag{lag}'] = m.groupby('household_key').sales_value.sum().reindex(hh).fillna(0.0)
    # long-run level and ratios
    g = tx.groupby('household_key')
    tot = g.sales_value.sum().reindex(hh).fillna(0.0)
    first = g.day.min().reindex(hh)
    tenure = (day - first).clip(lower=14)
    f['longrun_wk'] = tot / (tenure / 7.0)
    s28 = tx[tx.day > day - 28].groupby('household_key').sales_value.sum().reindex(hh).fillna(0.0)
    s84 = tx[tx.day > day - 84].groupby('household_key').sales_value.sum().reindex(hh).fillna(0.0)
    f['ratio28_lr'] = s28 / (f['longrun_wk'] * 4).clip(lower=1.0)
    f['ratio84_lr'] = s84 / (f['longrun_wk'] * 12).clip(lower=1.0)
    # basket value distribution, trailing 84d
    t84 = tx[tx.day > day - 84]
    b = t84.groupby(['household_key', 'basket_id']).sales_value.sum()
    bs = b.groupby('household_key').agg(['max', 'std', 'median'])
    f['basket_max_84'] = bs['max'].reindex(hh).fillna(0.0)
    f['basket_std_84'] = bs['std'].reindex(hh).fillna(0.0)
    f['basket_med_84'] = bs['median'].reindex(hh).fillna(0.0)
    # active days in last 28
    f['active_days_28'] = t84[t84.day > day - 28].groupby('household_key').day.nunique().reindex(hh).fillna(0.0)
    return f

feats = agent_api.build_features(make)
print(feats.shape)
e1 = agent_api.load_saved('e001_recent_behavior.parquet')
m = e1.merge(feats, on=['household_key', 'snapshot_day'], how='inner')
print("merged:", m.shape, "nulls:", int(m.isna().sum().sum()))
p = agent_api.save_table(m, 'e004_temporal.parquet')
print(p)
