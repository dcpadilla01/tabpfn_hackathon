
import pandas as pd, numpy as np
import agent_api as A

t = A.train_targets()
f = A.load_saved("e006_dynamics.parquet")
m = t.merge(f, on=["household_key","snapshot_day"], how="left")

cat_cols = ["classification_1","classification_2","classification_3","classification_4",
            "classification_5","homeowner_desc","kid_category_desc"]
num_cols = [c for c in f.columns if c not in ("household_key","snapshot_day")+tuple(cat_cols)]
num_cols = [c for c in num_cols if pd.api.types.is_numeric_dtype(f[c])]

ptr = m[m.snapshot_day<=375].dropna(subset=["spend_28"])
pv  = m[m.snapshot_day.isin([403,431])].dropna(subset=["spend_28"])
ytr2, yva2 = ptr.future_spend_4w.values, pv.future_spend_4w.values

Xn_tr = ptr[num_cols].astype(float); mu,sd = Xn_tr.mean(), Xn_tr.std().replace(0,1)
cols = {c: pd.get_dummies(ptr[c].astype(str).fillna("NA"), prefix=c[:6], dtype=float).columns for c in cat_cols}
def dm(df):
    Xn = ((df[num_cols].astype(float)-mu)/sd).fillna(0.0).values
    blocks=[Xn]
    for c in cat_cols:
        blocks.append(pd.get_dummies(df[c].astype(str).fillna("NA"), prefix=c[:6], dtype=float).reindex(columns=cols[c], fill_value=0).values)
    return np.hstack(blocks)
Xtr, Xva = dm(ptr), dm(pv)
w = np.linalg.solve(Xtr.T@Xtr + 100*np.eye(Xtr.shape[1]), Xtr.T@ytr2)
p = Xva@w
print("base pseudo-val MAE:", round(np.mean(np.abs(yva2-p)),3))

# multiplicative shrink
for s in [0.7,0.8,0.9,1.0,1.1]:
    print(f"  shrink x{s}: {np.mean(np.abs(yva2 - s*p)):.3f}")

# blend with spend_28 and with 0
for wt in [0.0,0.1,0.2,0.3,0.5]:
    print(f"  blend p*(1-w)+spend28*w (w={wt}): {np.mean(np.abs(yva2 - ((1-wt)*p + wt*pv.spend_28.values))):.3f}")

# error structure: over/under by y bucket
df = pd.DataFrame({"y":yva2,"p":p})
df["b"] = pd.qcut(df.y, [0,.25,.5,.75,.9,1.0], duplicates="drop")
print(df.groupby("b", observed=True).apply(lambda d: pd.Series(
    {"n":len(d),"y_med":d.y.median(),"p_med":d.p.median(),"bias":(d.p-d.y).mean(),"mae":np.abs(d.p-d.y).mean()}), include_groups=False).round(1).to_string())

# error structure by spend_28==0
z = pv.spend_28.values==0
print("\nspend_28==0 rows:", z.mean().round(3), " MAE:", round(np.mean(np.abs(yva2[z]-p[z])),2), " mean y:", round(yva2[z].mean(),2), " mean p:", round(p[z].mean(),2))
print("spend_28>0  rows: MAE:", round(np.mean(np.abs(yva2[~z]-p[~z])),2))
