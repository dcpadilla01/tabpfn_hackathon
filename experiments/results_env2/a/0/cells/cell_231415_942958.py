
import pandas as pd, numpy as np, agent_api, xgboost as xgb

feats = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = tt.merge(feats, on=['household_key','snapshot_day'], how='left')
FE = [c for c in feats.columns if c not in ('index','household_key','snapshot_day')]

def make_xy(d):
    X = d[FE].copy()
    for c in X.columns: X[c] = pd.to_numeric(X[c], errors='coerce')
    return X, d['future_spend_4w'].values

def qmodel(Xtr,ytr,alpha=0.5,depth=4,mcw=20,nest=600,lr=0.05,seed=0):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=depth,
                         min_child_weight=mcw, n_estimators=nest, learning_rate=lr,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=seed)
    m.fit(Xtr,ytr); return m

tr = df[df.snapshot_day<=403]; va = df[df.snapshot_day==431]
Xtr,ytr = make_xy(tr); Xv,yv = make_xy(va)
blv = Xv['exp4w_blend'].values

# robustness of alpha=0.45 across seeds and depths
for seed in [0,1,2]:
    for depth,mcw in [(4,20),(4,50),(5,15)]:
        pv = qmodel(Xtr,ytr,alpha=0.45,depth=depth,mcw=mcw,seed=seed).predict(Xv)
        print(f"a=.45 d{depth} mcw{mcw} s{seed}:", round(np.abs(np.clip(0.7*pv+0.3*blv,0,None)-yv).mean(),3))

# alpha sweep finer
for a in [0.40,0.42,0.44,0.46,0.48]:
    pv = qmodel(Xtr,ytr,alpha=a).predict(Xv)
    print(f"alpha={a}:", round(np.abs(np.clip(0.7*pv+0.3*blv,0,None)-yv).mean(),3))

# blend weight with alpha=0.45
p45 = qmodel(Xtr,ytr,alpha=0.45).predict(Xv)
for w in [0.5,0.6,0.7,0.8,0.9,1.0]:
    print(f"w={w}:", round(np.abs(np.clip(w*p45+(1-w)*blv,0,None)-yv).mean(),3))

# also check on second local holdout: 403 (train<=375)
tr2 = df[df.snapshot_day<=375]; va2 = df[df.snapshot_day==403]
X2,y2 = make_xy(tr2); Xv2,yv2 = make_xy(va2); blv2 = Xv2['exp4w_blend'].values
for a in [0.45,0.5]:
    pv2 = qmodel(X2,y2,alpha=a).predict(Xv2)
    print(f"holdout403 alpha={a}:", round(np.abs(np.clip(0.7*pv2+0.3*blv2,0,None)-yv2).mean(),3))
