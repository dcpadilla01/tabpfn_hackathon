names = ["e005_preds","e011_preds","e016_preds","e007_preds","e010_preds","e012_preds","e014_preds","e009_preds","e004_preds","e003_preds","e016_held","e016_allpreds"]
P = {}
for n in names:
    df = load_saved(f"{n}.parquet")
    P[n] = df
    print(n, df.shape, list(df.columns))

base = P["e005_preds"][["household_key","snapshot_day","prediction"]].rename(columns={"prediction":"e005"})
for n in names[1:10]:
    m = base.merge(P[n].rename(columns={"prediction":n}), on=["household_key","snapshot_day"])
    c = np.corrcoef(m.e005, m[n])[0,1]
    mad = np.abs(m.e005-m[n]).mean()
    print(n, "corr:", round(c,4), "mean|diff|:", round(mad,2))

oof = load_saved("oof_e5.parquet")
print("oof_e5:", oof.shape, list(oof.columns))
print(oof.head())
print(oof.snapshot_day.unique() if "snapshot_day" in oof.columns else "")
