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
