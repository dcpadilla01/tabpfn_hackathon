import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings("ignore")
tt = A.train_targets().sort_values(["household_key","snapshot_day"])
# expanding past-target stats per household (only targets whose window ends <= current snapshot day)
tt["win_end"] = tt.snapshot_day + 28
rows = []
for hk, g in tt.groupby("household_key"):
    ys = g.future_spend_4w.values; ds = g.snapshot_day.values; we = g.win_end.values
    for i in range(len(g)):
        past = (we[:i] <= ds[i])  # windows fully in the past
        pv = ys[:i][past]
        rows.append((hk, ds[i], len(pv), np.mean(pv) if len(pv) else np.nan,
                     np.median(pv) if len(pv) else np.nan,
                     np.std(pv) if len(pv)>1 else np.nan,
                     ys[i-1] if i>0 and we[i-1]<=ds[i] else np.nan))
pt = pd.DataFrame(rows, columns=["household_key","snapshot_day","n_past_tg","mean_past_tg","med_past_tg","std_past_tg","lag1_tg"])
print(pt.shape, pt.head(3).to_string())
print("coverage on rows:", pt.n_past_tg.notna().mean().round(3), "| mean n_past:", pt.n_past_tg.mean().round(2))
p = A.save_table(pt, "past_targets.parquet"); print("saved", p)
# MAE of simple past-median predictor on train rows with >=4 past targets
sub = pt[pt.n_past_tg>=4].merge(tt, on=["household_key","snapshot_day"])
print("rows:", len(sub), "MAE med_past_tg:", round(np.abs(sub.med_past_tg-sub.future_spend_4w).mean(),2),
      "MAE mean_past_tg:", round(np.abs(sub.mean_past_tg-sub.future_spend_4w).mean(),2))
