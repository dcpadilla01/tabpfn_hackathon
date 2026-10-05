
import agent_api, pandas as pd, numpy as np, xgboost as xgb

feats = agent_api.load_saved("feats_v4.parquet")
oof = agent_api.load_saved("oof_e013.parquet")
oof["resid"] = oof.y - oof.oof
oof["bin"] = pd.qcut(oof.oof, 10, duplicates="drop").astype(str)
corr = oof.groupby("bin").resid.median()
edges = np.unique(np.quantile(oof.oof, np.linspace(0, 1, 11)))

tt = agent_api.train_targets()
train = feats.merge(tt, on=["household_key","snapshot_day"])
val = feats[feats.snapshot_day >= 459].copy()
FE = [c for c in feats.columns if c not in ("household_key","snapshot_day")]
obj = [c for c in FE if str(train[c].dtype) == "category"]

def prep(df):
    X = df[FE].copy()
    for c in obj: X[c] = X[c].astype("category")
    return X

Xtr, ytr = prep(train), train.future_spend_4w.values
Xva = prep(val)
preds = []
for d, w in [(4,20),(5,40),(6,20),(5,60)]:
    for seed in (7, 13):
        m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, learning_rate=0.08,
                             n_estimators=400, n_jobs=4, tree_method="hist", enable_categorical=True,
                             random_state=seed, max_depth=d, min_child_weight=w)
        m.fit(Xtr, ytr)
        preds.append(m.predict(Xva))
p = np.mean(preds, axis=0)

bins = pd.cut(pd.Series(p), edges, include_lowest=True).astype(str)
adj = bins.map(corr).astype(float).fillna(0).values
p2 = np.clip(p + adj, 0, None)
r28 = val.spend_28.fillna(0).values
final = 0.97 * p2 + 0.03 * r28

out = pd.DataFrame({"household_key": val.household_key.values,
                    "snapshot_day": val.snapshot_day.values, "prediction": final})
print("rows:", len(out), "| mean %.1f median %.1f min %.1f max %.1f" % (final.mean(), np.median(final), final.min(), final.max()))
agent_api.save_table(out, "pred_e019.parquet")
