
import pandas as pd, numpy as np, agent_api

for name in ["e004_features","e005_newfeats","e005_preds","e004_preds","e003_preds"]:
    df = agent_api.load_saved(name+".parquet")
    print(name, df.shape)
    print(list(df.columns)[:40])
    print(df.head(3))
    print("---")


# ---- cell ----

import pandas as pd, numpy as np, agent_api
tt = agent_api.train_targets()
print(tt.shape, tt.columns.tolist())
print(tt.groupby("snapshot_day").future_spend_4w.agg(["count","mean","median"]).head(20))
print("frac zero:", (tt.future_spend_4w==0).mean())
print(tt.future_spend_4w.describe())


# ---- cell ----

import pandas as pd, numpy as np, agent_api

feats = agent_api.load_saved("e004_features.parquet")
tt = agent_api.train_targets()
df = feats.merge(tt, on=["household_key","snapshot_day"])
print(df.shape)

y = df.future_spend_4w.values

# how predictive is spend_28 alone (MAE-optimal affine)?
s28 = df.spend_28.values if "spend_28" in df.columns else None
print([c for c in df.columns if "spend" in c][:20])


# ---- cell ----

import pandas as pd, numpy as np, agent_api
from scipy.optimize import minimize

feats = agent_api.load_saved("e004_features.parquet")
tt = agent_api.train_targets()
df = feats.merge(tt, on=["household_key","snapshot_day"])
y = df.future_spend_4w.values
s28 = df.spend_28.values

def mae(p, y): return np.mean(np.abs(p-y))

# affine calibration of spend_28 -> y, minimizing MAE
def obj(theta):
    a,b = theta
    return mae(a+b*s28, y)
res = minimize(obj, x0=[np.median(y)-np.median(s28), 1.0], method="Nelder-Mead")
print("affine calib spend_28:", res.x, "MAE:", res.fun)
print("MAE raw spend_28:", mae(s28, y))
print("MAE pred median:", mae(np.full(len(y), np.median(y)), y))

# quantiles of y given zero vs positive
print("zero frac:", (y==0).mean(), "zero frac given s28==0:", (y[s28==0]==0).mean() if (s28==0).sum()>0 else None)

# residual after affine: correlation with other features
a,b = res.x
resid = y - (a+b*s28)
num_cols = [c for c in df.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
cors = df[num_cols].apply(lambda c: np.corrcoef(c.fillna(c.median()), resid)[0,1])
print(cors.sort_values(key=np.abs, ascending=False).head(20))


# ---- cell ----

import pandas as pd, numpy as np, agent_api
feats = agent_api.load_saved("e004_features.parquet")
print(list(feats.columns))
print(feats.snapshot_day.value_counts().sort_index())


# ---- cell ----

import pandas as pd, numpy as np, agent_api
feats = agent_api.load_saved("e005_newfeats.parquet")
print(list(feats.columns))
print(feats.head(3).T)


# ---- cell ----

import pandas as pd, numpy as np, agent_api
p5 = agent_api.load_saved("e005_preds.parquet")
p4 = agent_api.load_saved("e004_preds.parquet")
p3 = agent_api.load_saved("e003_preds.parquet")
m = p5.merge(p4, on=["household_key","snapshot_day"], suffixes=("_5","_4")).merge(p3, on=["household_key","snapshot_day"])
m = m.rename(columns={"prediction":"pred_3"})
print(m.describe())
print("corr 5 vs 4:", np.corrcoef(m.pred_5, m.pred_4)[0,1])
print("mean abs diff 5-4:", np.mean(np.abs(m.pred_5-m.pred_4)))
print("mean abs diff 5-3:", np.mean(np.abs(m.pred_5-m.pred_3)))

# per-snapshot means
print(m.groupby("snapshot_day")[["pred_5","pred_4","pred_3"]].mean())


# ---- cell ----

import pandas as pd, numpy as np, agent_api
p5 = agent_api.load_saved("e005_preds.parquet").rename(columns={"prediction":"pred_5"})
p4 = agent_api.load_saved("e004_preds.parquet").rename(columns={"prediction":"pred_4"})
p3 = agent_api.load_saved("e003_preds.parquet").rename(columns={"prediction":"pred_3"})
m = p5.merge(p4, on=["household_key","snapshot_day"]).merge(p3, on=["household_key","snapshot_day"])
print("corr 5v4:", np.corrcoef(m.pred_5, m.pred_4)[0,1], "corr 5v3:", np.corrcoef(m.pred_5, m.pred_3)[0,1])
print("mean abs diff 5-4:", np.mean(np.abs(m.pred_5-m.pred_4)))
print(m.groupby("snapshot_day")[["pred_5","pred_4","pred_3"]].mean())

# distribution of preds vs y
tt = agent_api.train_targets()
print(tt.future_spend_4w.describe())
print(m.pred_5.describe())


# ---- cell ----

import pandas as pd, numpy as np, agent_api, xgboost as xgb, time

feats = agent_api.load_saved("e004_features.parquet")
nf = agent_api.load_saved("e005_newfeats.parquet")
df = feats.merge(nf.drop(columns=["household_key","snapshot_day"]), left_index=True, right_index=True)
tt = agent_api.train_targets()
df = df.merge(tt, on=["household_key","snapshot_day"])
print(df.shape)

feat_cols = [c for c in df.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
X = df[feat_cols].astype(float).replace([np.inf,-np.inf], np.nan)
y = df.future_spend_4w.values
day = df.snapshot_day.values

def make_model(seed):
    return xgb.XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=0.5,
        n_estimators=1200, learning_rate=0.03, max_depth=6,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", n_jobs=8, random_state=seed, base_score=70.0)

t0=time.time()
# diagnostic: train on days<=403, predict 431
tr = day<=403; te = day==431
w = 0.5**((403-day[tr])/140.0)
m = make_model(0); m.fit(X[tr], y[tr], sample_weight=w)
p431 = m.predict(X[te]); y431 = y[te]
print("train time", time.time()-t0)
print("MAE 431:", np.mean(np.abs(p431-y431)))
res = y431-p431
print("median residual:", np.median(res), "mean:", res.mean())

# global shift
print("MAE after global median shift:", np.mean(np.abs(p431+np.median(res)-y431)))

# per-decile-bin median calibration
qs = np.quantile(p431, np.linspace(0,1,11))
bins = np.clip(np.digitize(p431, qs[1:-1]),0,9)
corr = np.array([np.median(res[bins==b]) for b in range(10)])
print("bin med resid:", np.round(corr,1))
print("MAE after bin calib:", np.mean(np.abs(p431+corr[bins]-y431)))

# zero-behavior: households with s28==0
s28_te = X[te][:, feat_cols.index("spend_28")]
print("n s28==0 at 431:", (s28_te==0).sum(), "median y:", np.median(y431[s28_te==0]), "median pred:", np.median(p431[s28_te==0]))
print("MAE on s28==0 subset raw:", np.mean(np.abs(p431[s28_te==0]-y431[s28_te==0])),
      "if pred=0:", np.mean(np.abs(y431[s28_te==0])))


# ---- cell ----

import pandas as pd, numpy as np, agent_api, xgboost as xgb

feats = agent_api.load_saved("e004_features.parquet")
nf = agent_api.load_saved("e005_newfeats.parquet")
df = feats.merge(nf.drop(columns=["household_key","snapshot_day"]), left_index=True, right_index=True)
tt = agent_api.train_targets()
df = df.merge(tt, on=["household_key","snapshot_day"])
feat_cols = [c for c in df.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
X = df[feat_cols].astype(float).replace([np.inf,-np.inf], np.nan)
y = df.future_spend_4w.values
day = df.snapshot_day.values

tr = day<=403; te = day==431
w = 0.5**((403-day[tr])/140.0)
m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5,
    n_estimators=1200, learning_rate=0.03, max_depth=6, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=8,
    random_state=0, base_score=70.0)
m.fit(X[tr], y[tr], sample_weight=w)
p431 = m.predict(X[te]); y431 = y[te]
s28_te = df.spend_28.values[te]

z = s28_te==0
print("MAE all:", np.mean(np.abs(p431-y431)))
print("MAE s28==0 raw:", np.mean(np.abs(p431[z]-y431[z])), " if pred=0:", np.mean(y431[z]))
print("n:", z.sum(), "median y z:", np.median(y431[z]), "mean pred z:", p431[z].mean())
print("dist y|z:", np.percentile(y431[z],[10,25,50,75,90]))
print("dist pred|z:", np.percentile(p431[z],[10,25,50,75,90]))
# what does pred look like for z households
print("frac pred>10 among z:", (p431[z]>10).mean())


# ---- cell ----

import pandas as pd, numpy as np, agent_api
feats = agent_api.load_saved("e004_features.parquet")
tt = agent_api.train_targets()
df = feats.merge(tt, on=["household_key","snapshot_day"])

# household-level consistency: same household across snapshots
g = df.groupby("household_key").future_spend_4w.agg(["mean","std","count"])
print("households with >=2 snapshots:", (g["count"]>=2).sum())
print("mean within-household std:", g.loc[g["count"]>=2,"std"].mean())
print("mean within-household std / overall std:", g.loc[g["count"]>=2,"std"].mean()/df.future_spend_4w.std())
# ICC-like: between-household variance share
m = df.future_spend_4w.mean()
bw = df.groupby("household_key").future_spend_4w.mean().var()
tot = df.future_spend_4w.var()
print("between-household var share:", bw/tot)
