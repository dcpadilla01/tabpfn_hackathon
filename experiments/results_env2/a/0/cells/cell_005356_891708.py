
import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor

f4 = agent_api.load_saved("feats_v4.parquet")
tt = agent_api.train_targets()
df = f4.merge(tt, on=["household_key","snapshot_day"])
y = df.future_spend_4w.values.astype(float)
print("y stats: zero frac", (y==0).mean().round(4), "| quantiles", np.quantile(y,[.1,.25,.5,.75,.9,.95,.99]).round(1))
for d in sorted(df.snapshot_day.unique()):
    yy = df[df.snapshot_day==d].future_spend_4w
    print(d, "n", len(yy), "zero%", round((yy==0).mean()*100,1), "median", round(yy.median(),1))

feats = [c for c in df.columns if c not in ["household_key","snapshot_day","future_spend_4w"]]
cat_cols = [c for c in feats if df[c].dtype == object]
X = pd.get_dummies(df[feats], columns=cat_cols, dummy_na=True)
tr = df.snapshot_day <= 403; va = df.snapshot_day == 431
ytr, yva = y[tr.values], y[va.values]
Xtr, Xva = X[tr.values], X[va.values]

m = XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=5, min_child_weight=40,
                 subsample=0.8, colsample_bytree=0.8, objective="reg:quantileerror",
                 quantile_alpha=0.5, random_state=0, n_jobs=4, tree_method="hist")
m.fit(Xtr, ytr)
p = np.clip(m.predict(Xva), 0, None)
print("\nday-431: zero frac", round((yva==0).mean(),3))
print("pred on y==0 rows: median", round(np.median(p[yva==0]),1), "mean", round(p[yva==0].mean(),1))
print("pred on y>0  rows: median", round(np.median(p[yva>0]),1))
print("MAE on y==0 rows", round(np.abs(p[yva==0]).mean(),2), "| share of total abs err",
      round(np.abs(p[yva==0]).sum()/np.abs(p-yva).sum(),3))
print("MAE on y>0 rows", round(np.abs(p[yva>0]-yva[yva>0]).mean(),2))
print("mean pred", round(p.mean(),1), "mean y", round(yva.mean(),1), "median pred", round(np.median(p),1), "median y", round(np.median(yva),1))
