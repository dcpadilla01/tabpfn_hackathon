
import pandas as pd, numpy as np
import agent_api as A

for name in ["e003_mktg.parquet","e004_seasonal.parquet","e006_dynamics.parquet","e007_staples.parquet"]:
    df = A.load_saved(name)
    print(name, df.shape)
    print(list(df.columns))
    print()

# correlations of e006 features with target on train rows
t = A.train_targets()
f = A.load_saved("e006_dynamics.parquet")
m = t.merge(f, on=["household_key","snapshot_day"], how="left")
num = [c for c in f.columns if c not in ("household_key","snapshot_day")]
num = [c for c in num if pd.api.types.is_numeric_dtype(m[c])]
cor = m[num].corrwith(m["future_spend_4w"]).sort_values(key=np.abs, ascending=False)
print("TARGET describe:"); print(m["future_spend_4w"].describe())
print("\nTop |corr| with target (e006 features):")
print(cor.head(40).round(3).to_string())
