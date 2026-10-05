
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor

F = load_saved("e004_features.parquet")
f5n = load_saved("e005_newfeats.parquet")
c5 = [c for c in f5n.columns if c not in ("household_key","snapshot_day")]
F = F.merge(f5n.drop(columns=[c for c in c5 if c in F.columns]), on=["household_key","snapshot_day"], how="inner")
F2 = load_saved("f_weekly.parquet").drop(columns=["index"], errors="ignore")
F = F.merge(F2, on=["household_key","snapshot_day"], how="left").fillna(F2.mean(numeric_only=True))
feat = [c for c in F.columns if c not in ("household_key","snapshot_day")]
tt = train_targets().set_index(["household_key","snapshot_day"]).future_spend_4w

def mat(days):
    X = F[F.snapshot_day.isin(days)].set_index(["household_key","snapshot_day"])[feat]
    return X, tt.reindex(X.index)

Xtr, ytr = mat([95,123,151,179,207,235,263,291,319,347,375])
w = pd.Series(0.5 ** ((403 - Xtr.index.get_level_values(1)) / 140.0), index=Xtr.index)
Xe, ye = mat([431]); Xc, yc = mat([403])

base = dict(learning_rate=0.03, n_estimators=2400, max_depth=6, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist", n_jobs=8)
m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **base)
m.fit(Xtr, ytr, sample_weight=w)
pe = pd.Series(m.predict(Xe), index=Xe.index).clip(lower=0)
pc = pd.Series(m.predict(Xc), index=Xc.index).clip(lower=0)
print("with weekly feats: MAE431 %.2f  MAE403 %.2f" % (np.abs(pe-ye).mean(), np.abs(pc-yc).mean()))

imp = pd.Series(m.feature_importances_, index=feat).sort_values(ascending=False)
print("top gains:"); print(imp.head(15).round(4).to_string())
print("weekly feats gain sum:", imp[[c for c in imp.index if c.startswith(('wk_','lag','sp','dow','wkend','topstore','ncommod'))]].sum().round(4))
