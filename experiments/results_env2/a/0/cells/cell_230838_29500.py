
import pandas as pd, numpy as np, agent_api, xgboost as xgb

feats = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = tt.merge(feats, on=['household_key','snapshot_day'], how='left')
FE = [c for c in feats.columns if c not in ('index','household_key','snapshot_day')]

def make_xy(d):
    X = d[FE].copy()
    for c in X.columns: X[c] = pd.to_numeric(X[c], errors='coerce')
    return X, d['future_spend_4w'].values

tr = df[df.snapshot_day<=403]; va = df[df.snapshot_day==431]
Xtr,ytr = make_xy(tr); Xva,yva = make_xy(va)
blend = Xva['exp4w_blend'].values

def qmodel(Xtr,ytr,alpha=0.5,depth=4,mcw=20,nest=600,lr=0.05):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=depth,
                         min_child_weight=mcw, n_estimators=nest, learning_rate=lr,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=0)
    m.fit(Xtr,ytr); return m

m = qmodel(Xtr,ytr); p = m.predict(Xva)
base = np.clip(0.7*p+0.3*blend,0,None)
print("base MAE:", np.abs(base-yva).mean())

tmp = pd.DataFrame({'y':yva,'e':np.abs(base-yva),'p':base,'dsl':Xva['days_since_last'].values})
tmp['dsl_b']=pd.cut(tmp.dsl,[0,7,14,28,56,100,10000])
print("\nMAE by days_since_last:")
print(tmp.groupby('dsl_b',observed=True).agg(n=('e','size'),mae=('e','mean'),mean_pred=('p','mean'),mean_y=('y','mean')).round(1))
z = tmp.y==0
print("\nzeros: n=",z.sum()," mean pred:",round(tmp.p[z].mean(),1)," MAE:",round(tmp.e[z].mean(),1)," err share:",round(tmp.e[z].sum()/tmp.e.sum()*100,1))

# --- idea A: two-part model: P(y>0) * median(y|y>0)
from xgboost import XGBClassifier
ycls = (ytr>0).astype(int)
clf = XGBClassifier(max_depth=4, min_child_weight=20, n_estimators=400, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=0, eval_metric='logloss')
clf.fit(Xtr,ycls)
pz = clf.predict_proba(Xva)[:,1]
m2 = qmodel(Xtr[ytr>0],ytr[ytr>0])
p2 = m2.predict(Xva)
two = np.clip(0.7*(pz*p2)+0.3*blend,0,None)
print("\nA two-part MAE:", np.abs(two-yva).mean())
for w in [0.5,0.7,0.9]:
    print(f"  two-part w={w}:", np.abs(np.clip(w*(pz*p2)+(1-w)*blend,0,None)-yva).mean())

# --- idea B: quantile on log(y+1), then expm1
m3 = qmodel(Xtr,np.log1p(ytr))
p3 = np.expm1(m3.predict(Xva))
print("\nB log-quantile MAE (w=0.7):", np.abs(np.clip(0.7*p3+0.3*blend,0,None)-yva).mean())
for w in [0.5,0.7,0.9,1.0]:
    print(f"  log w={w}:", np.abs(np.clip(w*p3+(1-w)*blend,0,None)-yva).mean())

# --- idea C: deeper/wider model to capture tail
m4 = qmodel(Xtr,ytr,depth=6,mcw=10,nest=900)
p4 = m4.predict(Xva)
print("\nC depth6 MAE:", np.abs(np.clip(0.7*p4+0.3*blend,0,None)-yva).mean())
