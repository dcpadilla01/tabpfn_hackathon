names = ["e005_preds","e011_preds","e016_preds","e007_preds","e010_preds","e012_preds","e014_preds","e009_preds","e004_preds","e003_preds"]
preds = {}
for n in names:
    df = load_saved(f"{n}.parquet")
    preds[n] = df
    print(n, df.shape, list(df.columns), round(df.prediction.mean(),2), round(df.prediction.std(),2))

import itertools
base = preds["e005_preds"][["household_key","snapshot_day","prediction"]].rename(columns={"prediction":"e005"})
print("base rows", base.shape)
for n in names[1:]:
    m = base.merge(preds[n][["household_key","snapshot_day","prediction"]], on=["household_key","snapshot_day"])
    c = np.corrcoef(m.prediction_x, m.prediction_y)[0,1]
    mad = np.abs(m.prediction_x-m.prediction_y).mean()
    print(n, "corr vs e005:", round(c,4), "mean|diff|:", round(mad,2))

oof = load_saved("oof_e5.parquet")
print("oof_e5:", oof.shape, list(oof.columns))
print(oof.head())
