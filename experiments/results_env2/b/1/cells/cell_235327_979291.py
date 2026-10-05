
import agent_api as A, pandas as pd, numpy as np
e8 = A.load_saved('e008_level_shape.parquet')
tt = A.train_targets()
m = tt.merge(e8, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values
def mae(p): return float(np.mean(np.abs(np.asarray(p)-y)))
print('pred0', round(mae(np.zeros(len(y))),2), 'const-median', round(mae(np.full(len(y), np.median(y))),2))
for c in ['sp28','sp84','sp364','sp728','z_med4w_hist','z_max4w_hist','sp28_rate','sp364_rate','splag1y','sp_lag1y','newma28','wksp_mean','tenure','days_since_last']:
    if c in m: print(c, round(mae(m[c].fillna(0).values),2))
print('blend sp28/med4w', round(mae(0.5*m['sp28'].fillna(0)+0.5*m['z_med4w_hist'].fillna(0)),2))
print('blend sp28/84/med', round(mae((m['sp28'].fillna(0)+m['sp84'].fillna(0)+m['z_med4w_hist'].fillna(0))/3),2))
# rank corr of all numeric features with target
num = [c for c in e8.columns if c not in ('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(m[c])]
ranks = m[num].rank()
yc = pd.Series(y).rank()
cor = {}
for c in num:
    s = ranks[c]; ok = s.notna()
    if ok.sum()>500: cor[c] = abs(float(np.corrcoef(s[ok], yc[ok])[0,1]))
top = sorted(cor.items(), key=lambda kv:-kv[1])
print('TOP30:', [(k, round(v,3)) for k,v in top[:30]])
print('BOT10:', [(k, round(v,3)) for k,v in top[-10:]])
print('per-snapshot target mean/median:')
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count']))
