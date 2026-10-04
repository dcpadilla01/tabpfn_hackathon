import numpy as np, pandas as pd
s = snapshot()
tx = s.transactions
wk = (tx.day + 8)//7
wsp = tx.assign(_w=wk).groupby('_w').sales_value.sum()
wsp = wsp[wsp.index>=5]  # drop onboarding weeks 1-4
# detrend: ratio to centered 13-week moving average
idx = wsp.index.values
ma = wsp.rolling(13, center=True, min_periods=5).mean()
ratio = wsp/ma
print("seasonal ratio by week (5..66):")
print(ratio.round(2).to_string())

train_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]
val_days = [459,487,515,543]
tt = train_targets()
daymean = tt.groupby('snapshot_day').future_spend_4w.mean()

def seasonal_index(w, mode='next4'):
    # year-ago analog: mean weekly spend in weeks w-51..w-48 (next 4 weeks a year ago)
    # relative to weeks w-55..w-52 (trailing 4 weeks a year ago)
    def meanw(ws):
        vals = [wsp.get(k, np.nan) for k in ws]
        vals = [v for v in vals if not np.isnan(v)]
        return np.mean(vals) if vals else np.nan
    nxt = meanw(range(w-51, w-47))
    trl = meanw(range(w-55, w-51))
    return nxt/trl if trl and not np.isnan(trl) and trl>0 else np.nan

rows=[]
for d in train_days:
    w = (d+8)//7
    # realized: population next-4-weeks spend vs trailing-4-weeks, using full data (analysis only)
    def meanw_real(ws):
        vals = [wsp.get(k, np.nan) for k in ws]
        return np.nanmean(vals)
    realized = meanw_real(range(w+1,w+5))/meanw_real(range(w-3,w+1))
    si = seasonal_index(w)
    rows.append((d, w, round(realized,3), round(si,3) if not np.isnan(si) else np.nan, round(daymean[d],1)))
df = pd.DataFrame(rows, columns=['day','week','realized_next4/trail4','yearago_index','target_mean'])
print(df.to_string())
print("\ncorr realized vs target_mean:", np.corrcoef(df.realized[3:], df.target_mean[3:])[0,1])
print("corr yearago_index vs target_mean:", np.corrcoef(df.yearago_index[8:], df.target_mean[8:])[0,1])
for d in val_days:
    w=(d+8)//7
    print("val day", d, "week", w, "yearago_index:", round(seasonal_index(w),3))