import pandas as pd, numpy as np, agent_api
t = agent_api.load_saved('e011_pruned_basket.parquet')
print('E011 table:', t.shape)
print('cols:', list(t.columns))
tt = agent_api.train_targets()
print('targets:', tt.shape)
g = tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','max', lambda s:(s==0).mean()])
g.columns=['mean','median','max','zero_share']
print(g.round(2))
m = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged:', m.shape)
num = [c for c in t.columns if c not in ('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(t[c])]
print('n numeric feats:', len(num))
cor = pd.Series({c: m[c].corr(m['future_spend_4w']) for c in num}).sort_values(key=np.abs, ascending=False)
print(cor.head(35).round(3).to_string())
print('weak (|corr|<0.02):', int((cor.abs()<0.02).sum()), 'of', len(cor))
# drift check: mean of a few key features per snapshot
for c in [x for x in num if ('28' in x and 'spend' in x.lower())][:4]:
    print(c, 'per-snapshot mean:', t.groupby('snapshot_day')[c].mean().round(1).to_dict())
print('target describe:', tt.future_spend_4w.describe().round(2).to_dict())


# ---- cell ----
import pandas as pd, numpy as np, agent_api
t = agent_api.load_saved('e011_pruned_basket.parquet')
tt = agent_api.train_targets()
tr_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]
va_days = [459,487,515,543]
num = [c for c in t.columns if c not in ('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(t[c])]
tr = t[t.snapshot_day.isin(tr_days)]; va = t[t.snapshot_day.isin(va_days)]
rows=[]
for c in num:
    a, b = tr[c].astype(float), va[c].astype(float)
    am, bm = np.nanmean(a), np.nanmean(b)
    asd = np.nanstd(a) + 1e-9
    # standardized mean shift
    shift = (bm-am)/asd
    # quantile overlap: fraction of val beyond train max / below train min
    hi = np.nanmean(b > np.nanmax(a)) if np.isfinite(np.nanmax(a)) else 0
    lo = np.nanmean(b < np.nanmin(a)) if np.isfinite(np.nanmin(a)) else 0
    rows.append((c, shift, hi, lo, am, bm))
d = pd.DataFrame(rows, columns=['feat','shift_std','val_above_trmax','val_below_trmin','tr_mean','va_mean']).set_index('feat')
print("Top drift by |shift|:")
print(d.reindex(d.shift_std.abs().sort_values(ascending=False).index).head(20).round(3).to_string())
print("\nTop by val>train-max out-of-range share:")
print(d.reindex(d.val_above_trmax.sort_values(ascending=False).index).head(15).round(3).to_string())
# correlation of drift with feature-target corr on train
m = t.merge(tt, on=['household_key','snapshot_day'])
mtr = m[m.snapshot_day.isin(tr_days)]
ct = mtr[num].corrwith(mtr.future_spend_4w)
d['train_corr'] = ct
d['risk'] = d.shift_std.abs()*d.train_corr.abs()
print("\nRisk = |shift|*|train corr|, top 15:")
print(d.reindex(d.risk.sort_values(ascending=False).index).head(15).round(3).to_string())


# ---- cell ----
import pandas as pd, numpy as np, agent_api
t = agent_api.load_saved('e011_pruned_basket.parquet')
drop = ['tenure','days_since_first','week','week_q',
        'spend_ly','ly_trail28','ly_future28',
        'total_spend','life_spend',
        'n_campaigns','camp_TypeA','camp_TypeB','camp_TypeC','camp_recent84',
        'camp_x_demo','campA_x_spend28',
        'redemp_total','days_since_redemp','z_active_wk_52']
drop = [c for c in drop if c in t.columns]
t2 = t.drop(columns=drop)
print('dropped', len(drop), '-> shape', t2.shape)
print('remaining cols:', len(t2.columns)-2)
p = agent_api.save_table(t2, 'e013_stationary.parquet')
print(p)
