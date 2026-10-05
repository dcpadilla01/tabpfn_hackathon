
import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor

f4 = agent_api.load_saved("feats_v4.parquet")
tt = agent_api.train_targets()
df = f4.merge(tt, on=["household_key","snapshot_day"])
feats = [c for c in df.columns if c not in ["household_key","snapshot_day","future_spend_4w"]]
cat_cols = [c for c in feats if df[c].dtype == object]
X = pd.get_dummies(df[feats], columns=cat_cols, dummy_na=True)
y = df.future_spend_4w.values.astype(float)
day = df.snapshot_day.values.astype(float)
tr = df.snapshot_day <= 403
va = df.snapshot_day == 431
blendv = df.loc[va,"exp4w_blend"].values

def train_eval(w=None, days=None, alpha=0.5, blend=0.0, seed=0, depth=5, mcw=40, lr=0.08, n=400):
    m = tr if days is None else (df.snapshot_day.isin(days))
    Xtr, ytr, dtr = X[m.values], y[m.values], day[m.values]
    if w is None:
        sw = np.ones(len(ytr))
    elif w == "recency":
        sw = np.clip((dtr - 95) / (403 - 95), 0.15, 1.0)
    elif w == "recency2":
        sw = np.clip((dtr - 95) / (403 - 95), 0.05, 1.0) ** 2
    mod = XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=depth, min_child_weight=mcw,
                       subsample=0.8, colsample_bytree=0.8, objective="reg:quantileerror",
                       quantile_alpha=alpha, random_state=seed, n_jobs=4, tree_method="hist")
    mod.fit(Xtr, ytr, sample_weight=sw)
    p = mod.predict(X[va.values])
    p = np.clip(p, 0, None)
    if blend > 0:
        p = (1-blend)*p + blend*blendv
    return np.abs(p - y[va.values]).mean()

print("base (E011-style)      ", round(train_eval(),3))
print("recency w              ", round(train_eval(w="recency"),3))
print("recency w^2            ", round(train_eval(w="recency2"),3))
print("drop day 95            ", round(train_eval(days=[123,151,179,207,235,263,291,319,347,375,403]),3))
print("drop 95, recency       ", round(train_eval(w="recency", days=[123,151,179,207,235,263,291,319,347,375,403]),3))
for b in [0.1,0.2,0.3]:
    print(f"base blend {b}          ", round(train_eval(blend=b),3))
for a in [0.48,0.52,0.54]:
    print(f"alpha {a}              ", round(train_eval(alpha=a),3))
print("deeper bag check d4/d6 ", round(train_eval(depth=4),3), round(train_eval(depth=6),3))
