
import pandas as pd, numpy as np

names = ["allF","e001_preds","e002_features","e002_preds","e003_preds","e004_features",
         "e004_new","e004_preds","e005_newfeats","e005_preds","e007_preds","e008_preds",
         "lagfeats","lagfeats2"]
store = {}
for nm in names:
    for arg in (nm, nm+".parquet"):
        try:
            df = load_saved(arg)
            store[nm] = df
            print(nm, df.shape, "|", list(df.columns)[:14])
            break
        except Exception as e:
            last = (arg, type(e).__name__, str(e)[:80])
    else:
        print(nm, "ERR", last)

print()
print("snapshot_days:", snapshot_days())
print("KEYS:", KEYS, "TARGET:", TARGET)

tt = train_targets()
print("train_targets:", tt.shape, tt[TARGET].describe().round(2).to_dict())

# correlation of saved val predictions
p4 = store.get("e004_preds"); p5 = store.get("e005_preds"); p3 = store.get("e003_preds")
if p4 is not None and p5 is not None:
    m = p4.merge(p5, on=["household_key","snapshot_day"], suffixes=("_4","_5"))
    val = m[m.snapshot_day.isin(snapshot_days()["validation"])]
    print("val rows:", len(val))
    print("corr e004 vs e005 (val):", val.prediction_4.corr(val.prediction_5).round(4))
    print("e005 val pred describe:", val.prediction_5.describe().round(2).to_dict())
    if p3 is not None:
        m = m.merge(p3, on=["household_key","snapshot_day"])
        m.rename(columns={"prediction":"pred_3"}, inplace=True)
        v = m[m.snapshot_day.isin(snapshot_days()["validation"])]
        print("corr e003 vs e005 (val):", v.pred_3.corr(v.prediction_5).round(4))
