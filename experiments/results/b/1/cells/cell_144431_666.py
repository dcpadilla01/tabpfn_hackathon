
import pandas as pd, numpy as np

e1 = agent_api.load_saved("e001_history.parquet")
print("E001 cols (%d):" % len(e1.columns), list(e1.columns))

tt = agent_api.train_targets()
print("\nTarget describe:\n", tt.future_spend_4w.describe())
print("zero frac:", round((tt.future_spend_4w==0).mean(),3))
print("\nBy snapshot day:\n", tt.groupby("snapshot_day").future_spend_4w.agg(["count","mean","median"]))

v = agent_api.snapshot(459)
tx = v.transactions
print("\ntx day range:", tx.day.min(), tx.day.max(), "week range:", tx.week_no.min(), tx.week_no.max())
wk = tx.groupby("week_no").sales_value.sum()
print("\nWeekly total spend (every 4th week):")
print(wk.iloc[::4].round(0))

print("\nCorr with future_spend_4w on train snapshots (lag364 = same 4wk window 1yr earlier):")
rows=[]
for d in [347, 375, 403, 431]:
    fut = tt[tt.snapshot_day==d].set_index("household_key").future_spend_4w
    l364 = tx[(tx.day>=d-363)&(tx.day<=d-336)].groupby("household_key").sales_value.sum()
    t28  = tx[(tx.day>=d-27)&(tx.day<=d)].groupby("household_key").sales_value.sum()
    t84  = tx[(tx.day>=d-83)&(tx.day<=d)].groupby("household_key").sales_value.sum()
    df = pd.DataFrame({"fut":fut,"lag364":l364,"t28":t28,"t84":t84}).fillna(0)
    est = df[df.lag364.notna()]
    rows.append((d, round(df.fut.corr(df.lag364),3), round(est.fut.corr(est.lag364),3),
                 round(df.fut.corr(df.t28),3), round(df.fut.corr(df.t84),3), len(df), est.shape[0]))
print("d, corr_all(lag364), corr_estab(lag364), corr(t28), corr(t84), n, n_estab:")
for r in rows: print(r)
