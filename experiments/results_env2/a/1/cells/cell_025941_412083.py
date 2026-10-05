import agent_api as api
import pandas as pd, numpy as np, xgboost as xgb, time
allF = api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = allF[feat_cols].astype(float); y = allF['future_spend_4w'].astype(float); days = allF['snapshot_day'].astype(int)
d431=(days==431); y431=y[d431].values; X431=X[d431]; f=allF[d431]
tr=(days>=151)&(days<=403)&y.notna()
w=0.5**(((403)-days[tr].values)/140)
m=xgb.XGBRegressor(n_estimators=2400, objective='reg:quantileerror', quantile_alpha=0.5, tree_method='hist',
    max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, nthread=-1, seed=1)
m.fit(X[tr],y[tr],sample_weight=w)
p=m.predict(X431)
# zero structure
z28 = f['spend_28'].values==0
z84 = f['spend_84'].values==0
print("frac spend_28==0: %.3f, actual target==0 among them: %.3f" % (z28.mean(), (y431[z28]==0).mean()))
print("frac spend_84==0: %.3f, actual target==0 among them: %.3f" % (z84.mean(), (y431[z84]==0).mean()))
print("model pred for spend_28==0: mean %.2f median %.2f, actual mean %.2f" % (p[z28].mean(), np.median(p[z28]), y431[z28].mean()))
print("MAE on z28 subset: model %.2f, zero-pred %.2f, actual-zero-frac %.2f" % (np.abs(p[z28]-y431[z28]).mean(), np.abs(y431[z28]).mean(), (y431[z28]==0).mean()))
# clipping thresholds
base=0.31*f['spend_84'].values
for thr in [0,5,10,15,20,30]:
    pc = p.copy(); pc[p<thr]=0
    print("clip thr %d: model MAE %.3f, blend MAE %.3f" % (thr, np.abs(pc-y431).mean(), np.abs(0.5*pc+0.5*base-y431).mean()))
# blend refinements
seqm=f['seq_mean'].values; s28=f['spend_28'].values
cands = {
 '0.5p+0.5b': 0.5*p+0.5*base,
 '0.45p+0.45b+0.1seq': 0.45*p+0.45*base+0.1*seqm,
 '0.5p+0.4b+0.1s28': 0.5*p+0.4*base+0.1*s28,
 '0.55p+0.35b+0.1s28': 0.55*p+0.35*base+0.1*s28,
 '0.5p+0.5b clip10': np.where(0.5*p+0.5*base<10,0,0.5*p+0.5*base),
 '0.5p+0.5b clip15': np.where(0.5*p+0.5*base<15,0,0.5*p+0.5*base),
 '0.5p+0.5b clip20': np.where(0.5*p+0.5*base<20,0,0.5*p+0.5*base),
}
for k,v in cands.items():
    print("%-22s MAE %.3f" % (k, np.abs(v-y431).mean()))
