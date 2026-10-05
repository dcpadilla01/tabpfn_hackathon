
import pandas as pd, numpy as np, agent_api, xgboost as xgb

feats = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = tt.merge(feats, on=['household_key','snapshot_day'], how='left')
FE = [c for c in feats.columns if c not in ('index','household_key','snapshot_day')]

def make_xy(d):
    X = d[FE].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    return X, d['future_spend_4w'].values

tr = df[df.snapshot_day<=403]; va = df[df.snapshot_day==431]
Xtr,ytr = make_xy(tr); Xva,yva = make_xy(va)

m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=4,
                     min_child_weight=20, n_estimators=600, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=0)
m.fit(Xtr,ytr)
p = m.predict(Xva)
blend = Xva['exp4w_blend'].values
final = np.clip(0.7*p+0.3*blend,0,None)

e = np.abs(final-yva)
print("MAE by target bucket:")
bins=[0,1,50,100,200,400,800,10000]
lab=pd.cut(yva,bins)
tmp=pd.DataFrame({'y':yva,'e':e,'p':final,'dsl':Xva['days_since_last'].values,'s84':Xva['spend_84'].values})
print(tmp.groupby(lab,observed=True).agg(n=('e','size'),mae=('e','mean'),mean_pred=('p','mean'),mean_y=('y','mean'),sum_e=('e','sum')).round(1))
print("\nshare of total abs error by bucket (%):")
print((tmp.groupby(lab,observed=True)['sum_e'].sum()/e.sum()*100).round(1))

print("\nMAE by days_since_last bucket:")
tmp['dsl_b']=pd.cut(tmp.dsl,[0,7,14,28,56,100,10000])
print(tmp.groupby('dsl_b',observed=True).agg(n=('e','size'),mae=('e','mean'),mean_pred=('p','mean'),mean_y=('y','mean')).round(1))

# how do zeros behave
z = tmp.y==0
print("\nzeros: n=",z.sum()," mean pred on zeros:",tmp.p[z].mean().round(1)," MAE on zeros:",tmp.e[z].mean().round(1))
nz = tmp.y>0
print("nonzeros: n=",nz.sum()," mean pred:",tmp.p[nz].mean().round(1)," mean y:",tmp.y[nz].mean().round(1)," MAE:",tmp.e[nz].mean().round(1))

# tune blend weight locally
for w in [0.5,0.6,0.7,0.8,0.9,1.0]:
    f=np.clip(w*p+(1-w)*blend,0,None)
    print(f"w={w}: MAE={np.abs(f-yva).mean():.3f}")
