
import pandas as pd, numpy as np
import agent_api as A

tabs = {}
for name in ["e001_history", "e007_new", "e008_decomp2"]:
    df = A.load_saved(name + ".parquet")
    tabs[name] = df
    print("==", name, df.shape)
    print(list(df.columns))
    print()

tt = A.train_targets()
print("targets:", tt.shape)
print(tt["future_spend_4w"].describe())
print("zero frac:", (tt["future_spend_4w"] == 0).mean())
print()

e8, e1, e7 = tabs["e008_decomp2"], tabs["e001_history"], tabs["e007_new"]
c8 = set(e8.columns); c1 = set(e1.columns); c7 = set(e7.columns)
print("e001 cols not in e008:", sorted(c1 - c8))
print("e007 cols not in e008:", sorted(c7 - c8))
print()

m = tt.merge(e8, on=["household_key", "snapshot_day"], how="inner")
print("merged e8+target:", m.shape)
num = [c for c in e8.columns if c not in ("household_key", "snapshot_day") and pd.api.types.is_numeric_dtype(m[c])]
corr = m[num].corrwith(m["future_spend_4w"]).abs().sort_values(ascending=False)
print("top |corr| with target:")
print(corr.head(20))
print()

# naive predictor MAEs on train rows
def mae(p, y):
    return np.mean(np.abs(p - y))
y = m["future_spend_4w"].values
for c in num:
    if m[c].notna().sum() > 0.9 * len(m):
        pass
cands = [c for c in num if any(k in c.lower() for k in ["spend", "usual", "p_active", "exp", "pred"])]
for c in cands[:25]:
    v = m[c].values
    print(f"{c:35s} MAE={mae(v, y):8.3f}  corr={np.corrcoef(v, y)[0,1]:.3f}")
