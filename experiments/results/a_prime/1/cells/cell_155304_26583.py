import agent_api as A
import pandas as pd, numpy as np

e3 = A.load_saved('e010_l13fix.parquet')
tt = A.train_targets()
df = e3.merge(tt, on=['household_key','snapshot_day'], how='left')
tr = df[df.future_spend_4w.notna()].copy()
# only rows with REAL l13
sub = tr[tr.has_real_l13==1]
print('n real-l13 train rows:', len(sub))
print('corr l13 vs y:', round(float(sub.spend_l13.corr(sub.future_spend_4w)),3))
w = np.polyfit(sub.spend_l13, sub.future_spend_4w, 1)
print('linear fit: y =', round(w[0],3), '* l13 +', round(w[1],2))
p = np.polyval(w, sub.spend_l13)
print('alone MAE on these rows:', round(float(np.abs(p-sub.future_spend_4w).mean()),2))
print('MAE of spend_l123_mean on same rows:', round(float(np.abs(sub.spend_l123_mean-sub.future_spend_4w).mean()),2))
# ratio feature distribution (clip the crazy tail)
r = sub.l13_over_recent.clip(0,5)
print('l13_over_recent (clipped 0-5) quantiles:', np.percentile(r,[10,25,50,75,90]).round(2))
print('corr ratio vs y:', round(float(r.corr(sub.future_spend_4w)),3))
print('corr ratio vs spend_l123_mean:', round(float(r.corr(sub.spend_l123_mean)),3))
print('corr l13 vs l123_mean:', round(float(sub.spend_l13.corr(sub.spend_l123_mean)),3))
