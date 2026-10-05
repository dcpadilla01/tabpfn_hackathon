import pandas as pd, numpy as np, xgboost as xgb, time
F = agent_api.load_saved('allF.parquet'); tt = agent_api.train_targets()
FEAT = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = tt.merge(F.drop(columns=['future_spend_4w']), on=['household_key','snapshot_day'], how='left')
use_days = [95,123,151,179,207,235,263,291,319,347,375]
w = 0.5 ** ((431 - tr.snapshot_day)/140.0); m = tr.snapshot_day.isin(use_days)
Xw, yw, sw = tr.loc[m,FEAT], tr.loc[m,'future_spend_4w'].values, w.loc[m].values
common = dict(n_estimators=1200, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=10, reg_lambda=5.0, tree_method='hist', n_jobs=8, random_state=7)
t0=time.time()
mq = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, **common).fit(Xw, yw, sample_weight=sw)
ml = xgb.XGBRegressor(objective='reg:squarederror', **common).fit(Xw, np.log1p(yw), sample_weight=sw)
mc = xgb.XGBClassifier(objective='binary:logistic', **common).fit(Xw, (yw>0).astype(int), sample_weight=sw)
ma = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, **common).fit(Xw[yw>0], yw[yw>0], sample_weight=sw[yw>0])
print('fits done %.0fs' % (time.time()-t0))
Pq = mq.predict(F[FEAT]); Pl = np.expm1(ml.predict(F[FEAT])); Pc = mc.predict_proba(F[FEAT])[:,1]; Pa = ma.predict(F[FEAT])
P = F[['household_key','snapshot_day']].copy(); P['pq']=Pq; P['pl']=Pl; P['pc']=Pc; P['pa']=Pa
agent_api.save_table(P, 'e016_allpreds.parquet')
# held-out eval on 403/431
h = tr[tr.snapshot_day.isin([403,431])].merge(P, on=['household_key','snapshot_day'], how='left')
y = h.future_spend_4w.values
def mae(p): return np.abs(p-y).mean()
print('held MAE: quantile %.2f | logL2 %.2f | actquant %.2f' % (mae(h.pq), mae(h.pl), mae(h.pa)))
print('held mean act %.1f | pq %.1f pl %.1f pa %.1f' % (y.mean(), h.pq.mean(), h.pl.mean(), h.pa.mean()))
for wq in [0.0,0.3,0.5,0.7,1.0]:
    b = wq*h.pq + (1-wq)*h.pl
    print('  arith pq*wq+pl*(1-wq) wq=%.1f -> %.2f' % (wq, mae(b.values)))
for wq in [0.3,0.5,0.7]:
    g = np.exp(wq*np.log(np.clip(h.pq,1e-3,None)) + (1-wq)*np.log(np.clip(h.pl,1e-3,None)))
    print('  geo wq=%.1f -> %.2f' % (wq, mae(g)))
# two-part: p<0.5 -> 0 else pa
tp = np.where(h.pc < 0.5, 0.0, h.pa)
print('two-part (0 if p<.5 else pa) -> %.2f  (p<.5 frac %.2f)' % (mae(tp), (h.pc<0.5).mean()))
tp2 = np.where(h.pc < 0.4, 0.0, h.pa)
print('two-part thr .4 -> %.2f (frac %.2f)' % (mae(tp2), (h.pc<0.4).mean()))
# blend quantile with two-part
for wq in [0.3,0.5,0.7]:
    print('  pq*wq + tp*(1-wq) wq=%.1f -> %.2f' % (wq, mae(wq*h.pq + (1-wq)*tp)))
# blend quantile with log
print('corr pq pl: %.3f' % h.pq.corr(h.pl))