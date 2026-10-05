ap = load_saved("e016_allpreds.parquet")
print(ap.snapshot_day.value_counts().sort_index())
held = load_saved("e016_held.parquet")
print(held.snapshot_day.value_counts())
print(held.head())

# merge oof_e5 with allpreds on train rows
oof = load_saved("oof_e5.parquet")
m = oof.merge(ap, on=["household_key","snapshot_day"], how="inner")
print("merged train rows:", m.shape)
y = m.future_spend_4w.values
def mae(p): return np.abs(p-y).mean()
print("e5 OOF MAE:", round(mae(m.pred),3))
for c in ["pq","pl","pc","pa"]:
    print(c, "OOF MAE:", round(mae(m[c].values),3), "mean:", round(m[c].mean(),1), "std:", round(m[c].std(),1))
print("target mean:", round(y.mean(),1), "std:", round(y.std(),1))
