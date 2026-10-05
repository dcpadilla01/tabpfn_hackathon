
import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor

f4 = agent_api.load_saved("feats_v4.parquet")
tt = agent_api.train_targets()
df = f4.merge(tt, on=["household_key","snapshot_day"])
y = df.future_spend_4w.values.astype(float)
feats = [c for c in df.columns if c not in ["household_key","snapshot_day","future_spend_4w"]]
cat_cols = [c for c in feats if df[c].dtype == object]
X = pd.get_dummies(df[feats], columns=cat_cols, dummy_na=True)
tr = df.snapshot_day <= 403; va = df.snapshot_day == 431
ytr, yva = y[tr.values], y[va.values]
Xtr, Xva = X[tr.values], X[va.values]
blendv = df.loc[va,"exp4w_blend"].values

def fit(obj="reg:quantileerror", alpha=0.5, depth=5, mcw=40, seed=0, n=400, lr=0.08):
    m = XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=depth, min_child_weight=mcw,
                     subsample=0.8, colsample_bytree=0.8, objective=obj,
                     quantile_alpha=alpha if "quantile" in obj else None,
                     random_state=seed, n_jobs=4, tree_method="hist")
    m.fit(Xtr, ytr)
    return np.clip(m.predict(Xva), 0, None)

# A) quantile + L1 average
p_q = fit(); p_l1 = fit(obj="reg:absoluteerror")
for w in [0.25,0.5]:
    pm = (1-w)*p_q + w*p_l1
    print(f"avg quantile+L1 w={w}: MAE", round(np.abs(pm-yva).mean(),3))

# B) two-stage: zero classifier + quantile regressor on positives
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
pzero_tr = (ytr == 0).astype(int)
gbc = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=0)
gbc.fit(Xtr, pzero_tr)
pz = gbc.predict_proba(Xva)[:,1]
from sklearn.metrics import roc_auc_score
print("\nzero-classifier AUC on 431:", round(roc_auc_score((yva==0).astype(int), pz),4))
pos = ytr > 0
pq_pos = fit(ytr_=ytr[pos])
best = None
for thr in [0.35,0.45,0.55,0.65]:
    pm = np.where(pz > thr, 0.0, pq_pos)
    mae = np.abs(pm-yva).mean()
    print(f"two-stage thr={thr}: MAE {mae:.3f}")
    if best is None or mae < best[0]: best = (mae, thr)
print("best two-stage:", best)

# C) simple calibrated shrink: p' = a + b*p fit by least squares on inner val
A = np.vstack([np.ones_like(p_q), p_q]).T
coef, *_ = np.linalg.lstsq(A, yva, rcond=None)
print("calibration a,b on 431:", coef.round(4), "-> in-sample MAE", round(np.abs(A@coef - yva).mean(),3))
