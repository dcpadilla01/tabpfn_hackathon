import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
y = tt["future_spend_4w"]
print("target rows:", len(tt), "| snapshots:", sorted(tt.snapshot_day.unique()))
print(y.describe().round(2).to_dict())
print("zero share:", round((y==0).mean(),4))

names = ["e009_macro.parquet","e008_decomp2.parquet","e001_history.parquet","e011_display.parquet","e012_full.parquet"]
tabs = {}
for nm in names:
    try:
        df = A.load_saved(nm)
        tabs[nm] = df
        print("\n==", nm, df.shape)
        print(list(df.columns))
    except Exception as e:
        print(nm, "ERR", repr(e))

def corrs(df, label, k=50):
    m = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
    rows = []
    for c in df.columns:
        if c in ("household_key","snapshot_day"): continue
        x = m[c]
        ok = x.notna() & np.isfinite(x) if pd.api.types.is_numeric_dtype(x) else x.notna()
        if ok.sum() < 50: 
            rows.append((c, np.nan, np.nan, ok.mean())); continue
        if pd.api.types.is_numeric_dtype(x):
            r = np.corrcoef(x[ok].astype(float), y[ok].astype(float))[0,1]
            s = pd.Series(x[ok].astype(float)).corr(pd.Series(y[ok].astype(float)), method="spearman")
        else:
            r = np.nan; s = np.nan
        rows.append((c, r, s, ok.mean()))
    res = pd.DataFrame(rows, columns=["feat","pearson","spearman","nonnull"])
    res["absP"] = res.pearson.abs()
    res = res.sort_values("absP", ascending=False)
    print(f"\n--- correlations with target: {label} (merged {m.shape}) ---")
    print(res.head(k).round(3).to_string(index=False))
    return res

c9 = corrs(tabs["e009_macro.parquet"], "E009")
c1 = corrs(tabs["e001_history.parquet"], "E001", k=15)
