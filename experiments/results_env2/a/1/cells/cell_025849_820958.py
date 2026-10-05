import agent_api as api
import pandas as pd, numpy as np
pd.set_option('display.width', 250)
allF = api.load_saved('allF.parquet')
tt = api.train_targets()
m = tt.merge(allF.drop(columns=['future_spend_4w']), on=['household_key','snapshot_day'])
# understand seas_ok
for s in [403, 431]:
    sub = m[m.snapshot_day==s]
    print(s, "n", len(sub), "seas_ok mean %.2f" % sub.seas_ok.mean(),
          "tenure>=364 frac %.2f" % (sub.tenure>=364).mean(),
          "spend_seas_364>0 frac %.2f" % (sub.spend_seas_364>0).mean(),
          "spend_lag1y>0 frac %.2f" % (sub.spend_lag1y>0).mean())
sub = m[(m.snapshot_day==431)]
s1 = sub[sub.seas_ok==1]
yt = s1['future_spend_4w'].values
print("\n@431 seas_ok==1 n=%d target mean %.1f" % (len(s1), yt.mean()))
for c in ['spend_seas_364','spend_seas_336','spend_lag1y','spend_84','spend_28','seq_mean']:
    print("  %s: corr %.3f MAE %.2f" % (c, np.corrcoef(s1[c],yt)[0,1], np.abs(s1[c].values-yt).mean()))
print("  0.31*spend_84 MAE %.2f" % np.abs(0.31*s1['spend_84'].values-yt).mean())
print("  0.5*(seas364+seas336) MAE %.2f" % np.abs(0.5*(s1['spend_seas_364']+s1['spend_seas_336']).values-yt).mean())
print("  mean(seas364,seas336,0.31*84) MAE %.2f" % np.abs(((s1['spend_seas_364']+s1['spend_seas_336'])/2*0.5+0.31*s1['spend_84']*0.5).values-yt).mean())
# and on the seas_ok==0 subset, baseline behavior
s0 = sub[sub.seas_ok==0]
yt0 = s0['future_spend_4w'].values
print("@431 seas_ok==0 n=%d: 0.31*84 MAE %.2f, spend_28 MAE %.2f" % (len(s0), np.abs(0.31*s0['spend_84'].values-yt0).mean(), np.abs(s0['spend_28'].values-yt0).mean()))
# tenure distribution overall
print("\ntenure describe:"); print(allF['tenure'].describe().round(0))
print("frac tenure>=364 by snapshot:")
print(allF.groupby('snapshot_day')['tenure'].apply(lambda x:(x>=364).mean()).round(2))
