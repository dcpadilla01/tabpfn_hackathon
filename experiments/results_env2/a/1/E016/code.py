import pandas as pd, numpy as np
tt = agent_api.train_targets()
print('targets', tt.shape)
y = tt.future_spend_4w.values
print('mean %.2f med %.2f zero %.3f' % (y.mean(), np.median(y), (y==0).mean()))
print('q', np.round(np.quantile(y,[.05,.1,.25,.5,.75,.9,.95,.99]),1))
F = agent_api.load_saved('allF.parquet')
print('allF', F.shape)
print('days', sorted(F.snapshot_day.unique()))
print('cols', F.shape[1])
print(list(F.columns))
p5 = agent_api.load_saved('e005_preds.parquet')
print('e005_preds', p5.shape, sorted(p5.snapshot_day.unique()))
print(p5.head(3))


# ---- cell ----
import pandas as pd, numpy as np
for t in ['e004_new.parquet','e005_newfeats.parquet','lagfeats.parquet','lagfeats2.parquet','f_weekly.parquet','repro_e5.parquet']:
    df = agent_api.load_saved(t)
    print(t, df.shape)
    print(' ', [c for c in df.columns if c not in ('household_key','snapshot_day')][:40])


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, time
F = agent_api.load_saved('allF.parquet')
tt = agent_api.train_targets()
train_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]
val_days = [459,487,515,543]
FEAT = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
print('n_feat', len(FEAT))
# per-snapshot target stats
g = tt.groupby('snapshot_day').future_spend_4w.agg(['mean','median',lambda s:(s==0).mean()])
g.columns=['mean','median','zerofrac']; print(g.round(1))
p5 = agent_api.load_saved('e005_preds.parquet')
gp = p5.groupby('snapshot_day').prediction.agg(['mean','median',lambda s:(s<=1).mean()])
gp.columns=['mean','median','nearzero']; print(gp.round(1))
print(xgb.__version__)

# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, time
F = agent_api.load_saved('allF.parquet')
tt = agent_api.train_targets()
FEAT = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = tt.merge(F.drop(columns=['future_spend_4w']), on=['household_key','snapshot_day'], how='left')
tr_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]
use_days = tr_days[:-2]
ref = 431
w = 0.5 ** ((ref - tr.snapshot_day)/140.0)
m = tr.snapshot_day.isin(use_days)
X = tr.loc[m, FEAT]; y = tr.loc[m,'future_spend_4w'].values; sw = w.loc[m].values
t0=time.time()
model = xgb.XGBRegressor(objective='reg:quantile', quantile_alpha=0.5, n_estimators=1200,
                         learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=10, reg_lambda=5.0, tree_method='hist', n_jobs=8)
model.fit(X, y, sample_weight=sw)
print('fit %.1fs' % (time.time()-t0))
tr['pred'] = model.predict(tr[FEAT])
oof = tr[tr.snapshot_day.isin(use_days)]
err = (oof.pred - oof.future_spend_4w).abs()
print('OOF MAE %.3f' % err.mean())
print(oof.groupby('snapshot_day').apply(lambda d: (d.pred-d.future_spend_4w).abs().mean(), include_groups=False).round(1).to_dict())
oof2 = oof.copy(); oof2['dec'] = pd.qcut(oof2.pred, 10, duplicates='drop')
cal = oof2.groupby('dec', observed=True).apply(lambda d: pd.Series({'pred':d.pred.mean(),'act':d.future_spend_4w.mean(),'n':len(d),'mae':(d.pred-d.future_spend_4w).abs().mean()}), include_groups=False)
print(cal.round(1))
agent_api.save_table(oof[['household_key','snapshot_day','pred','future_spend_4w']], 'oof_e5.parquet')

# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, time
F = agent_api.load_saved('allF.parquet')
tt = agent_api.train_targets()
FEAT = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = tt.merge(F.drop(columns=['future_spend_4w']), on=['household_key','snapshot_day'], how='left')
tr_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]
use_days = tr_days[:-2]
ref = 431
w = 0.5 ** ((ref - tr.snapshot_day)/140.0)
m = tr.snapshot_day.isin(use_days)
X = tr.loc[m, FEAT]; y = tr.loc[m,'future_spend_4w'].values; sw = w.loc[m].values
t0=time.time()
model = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, n_estimators=1200,
                         learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=10, reg_lambda=5.0, tree_method='hist', n_jobs=8)
model.fit(X, y, sample_weight=sw)
print('fit %.1fs' % (time.time()-t0))
tr['pred'] = model.predict(tr[FEAT])
oof = tr[tr.snapshot_day.isin(use_days)]
err = (oof.pred - oof.future_spend_4w).abs()
print('OOF MAE %.3f' % err.mean())
print(oof.groupby('snapshot_day').apply(lambda d: (d.pred-d.future_spend_4w).abs().mean(), include_groups=False).round(1).to_dict())
oof2 = oof.copy(); oof2['dec'] = pd.qcut(oof2.pred, 10, duplicates='drop')
cal = oof2.groupby('dec', observed=True).apply(lambda d: pd.Series({'pred':d.pred.mean(),'act':d.future_spend_4w.mean(),'n':len(d),'mae':(d.pred-d.future_spend_4w).abs().mean()}), include_groups=False)
print(cal.round(1))
agent_api.save_table(oof[['household_key','snapshot_day','pred','future_spend_4w']], 'oof_e5.parquet')

# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, time
F = agent_api.load_saved('allF.parquet')
tt = agent_api.train_targets()
FEAT = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = tt.merge(F.drop(columns=['future_spend_4w']), on=['household_key','snapshot_day'], how='left')
tr_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]
use_days = tr_days[:-2]
w = 0.5 ** ((431 - tr.snapshot_day)/140.0)
m = tr.snapshot_day.isin(use_days)
model = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, n_estimators=1200,
                         learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=10, reg_lambda=5.0, tree_method='hist', n_jobs=8)
model.fit(tr.loc[m,FEAT], tr.loc[m,'future_spend_4w'].values, sample_weight=w.loc[m].values)
tr['pred'] = model.predict(tr[FEAT])
# held-out late snapshots 403, 431: bias check
for d in [375, 403, 431]:
    s = tr[tr.snapshot_day==d]
    print(d, 'n=%d predmean %.1f actmean %.1f bias %.1f MAE %.1f' % (len(s), s.pred.mean(), s.future_spend_4w.mean(), s.pred.mean()-s.future_spend_4w.mean(), (s.pred-s.future_spend_4w).abs().mean()))
# multiplicative/additive correction search on 403+431 held out
h = tr[tr.snapshot_day.isin([403,431])]
base = (h.pred-h.future_spend_4w).abs().mean()
print('held403+431 base MAE %.2f' % base)
for c in [0.0,2.5,5,7.5,10,12.5,15,20]:
    print('  +%.1f -> %.2f' % (c, (h.pred+c-h.future_spend_4w).abs().mean()))
for f in [1.0,1.05,1.1,1.15,1.2,1.25]:
    print('  x%.2f -> %.2f' % (f, (h.pred*f-h.future_spend_4w).abs().mean()))
# feature drift: mean spend_28 by snapshot (all 17 snaps incl val)
g = F.groupby('snapshot_day').spend_28.mean()
print('spend_28 mean by day:'); print(g.round(1).to_dict())
g2 = F.groupby('snapshot_day').spend_84.mean()
print('spend_84 mean by day:'); print(g2.round(1).to_dict())

# ---- cell ----
import pandas as pd, numpy as np
oof = agent_api.load_saved('oof_e5.parquet')
oof['resid'] = oof.future_spend_4w - oof.pred
print('OOF median(act-pred) = %.2f  mean = %.2f' % (oof.resid.median(), oof.resid.mean()))
oof['dec'] = pd.qcut(oof.pred, 10, duplicates='drop')
med = oof.groupby('dec', observed=True).agg(pred=('pred','mean'), act_med=('future_spend_4w','median'), act_mean=('future_spend_4w','mean'), resid_med=('resid','median'), n=('resid','size'))
print(med.round(1))
# global shift transfer test on held-out 403/431
F = agent_api.load_saved('allF.parquet'); tt = agent_api.train_targets()
tr = tt.merge(F.drop(columns=['future_spend_4w']), on=['household_key','snapshot_day'], how='left')
# recompute preds for 403/431 using saved e5 preds? e5 preds are val only. Use oof mapping applied to held-out preds:
# first retrain quickly to get preds at 403/431 (same config as before)
import xgboost as xgb, time
FEAT = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
use_days = [95,123,151,179,207,235,263,291,319,347,375]
w = 0.5 ** ((431 - tr.snapshot_day)/140.0); m = tr.snapshot_day.isin(use_days)
model = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, n_estimators=1200, learning_rate=0.03,
    max_depth=6, subsample=0.8, colsample_bytree=0.8, min_child_weight=10, reg_lambda=5.0, tree_method='hist', n_jobs=8)
t0=time.time(); model.fit(tr.loc[m,FEAT], tr.loc[m,'future_spend_4w'].values, sample_weight=w.loc[m].values); print('fit %.0fs'%(time.time()-t0))
tr['pred'] = model.predict(tr[FEAT])
h = tr[tr.snapshot_day.isin([403,431])].copy()
print('held base MAE %.2f' % (h.pred-h.future_spend_4w).abs().mean())
print('held median(act-pred) = %.2f' % (h.future_spend_4w-h.pred).median())
# apply OOF per-decile median mapping (bin edges from OOF pred)
bins = oof.groupby('dec', observed=True)['pred'].mean().values
edges = np.unique(pd.qcut(oof.pred, 10, duplicates='drop', retbins=True)[1])
h['bin'] = pd.cut(h.pred, edges, include_lowest=True)
mapping = oof.groupby('dec', observed=True).apply(lambda d: d.future_spend_4w.median() - d.pred.mean(), include_groups=False)
h['map_pred'] = h.pred + h['bin'].map(mapping).fillna(0)
print('held MAE after per-decile median mapping: %.2f' % (h.map_pred-h.future_spend_4w).abs().mean())
for c in [0,5,10,15,20,25]:
    print('  held +%.0f -> %.2f' % (c, (h.pred+c-h.future_spend_4w).abs().mean()))
# also OOF MAE after mapping
oof['map_pred'] = oof.pred + oof['dec'].map(mapping).fillna(0)
print('OOF MAE after mapping: %.2f (base %.2f)' % ((oof.map_pred-oof.future_spend_4w).abs().mean(), (oof.pred-oof.future_spend_4w).abs().mean()))

# ---- cell ----
import pandas as pd, numpy as np
r = agent_api.load_saved('repro_e5.parquet')
print(r.shape, list(r.columns))
print(r.head())
p5 = agent_api.load_saved('e005_preds.parquet'); p11 = agent_api.load_saved('e011_preds.parquet')
m = p5.merge(p11, on=['household_key','snapshot_day'], suffixes=('_e5','_e11'))
print('corr e5 vs e11 preds: %.4f' % m.prediction_e5.corr(m.prediction_e11))
print('mean diff', (m.prediction_e5-m.prediction_e11).abs().mean())
if 'p13' in r.columns and 'p11' in r.columns:
    print('repro p13 vs p11 corr %.4f' % r.p13.corr(r.p11))
# blend check (no scoring, just sanity of spread)
for w in [0,0.25,0.5,0.75,1.0]:
    b = w*m.prediction_e5 + (1-w)*m.prediction_e11
    print('w_e5=%.2f blend mean %.1f std %.1f' % (w, b.mean(), b.std()))

# ---- cell ----
import pandas as pd, numpy as np
# E016: multiplicative blend of quantile-XGB (E005) and squared-error XGB on same features
F = agent_api.load_saved('allF.parquet'); tt = agent_api.train_targets()
FEAT = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = tt.merge(F.drop(columns=['future_spend_4w']), on=['household_key','snapshot_day'], how='left')
use_days = [95,123,151,179,207,235,263,291,319,347,375]
w = 0.5 ** ((431 - tr.snapshot_day)/140.0); m = tr.snapshot_day.isin(use_days)
Xw, yw = tr.loc[m,FEAT], tr.loc[m,'future_spend_4w'].values, w.loc[m].values
t0=time.time()
# squared-error model (3 seeds for stability)
sq_preds = []
for seed in [7,17,27]:
    msq = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1200, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10, reg_lambda=5.0, tree_method='hist', n_jobs=8, random_state=seed)
    msq.fit(Xw, yw, sample_weight=sw)
    sq_preds.append(msq.predict(F[FEAT]))
    print('seed', seed, 'done %.0fs' % (time.time()-t0))
P = F[['household_key','snapshot_day']].copy()
for i,p in enumerate(sq_preds): P['sq%d'%i] = p
agent_api.save_table(P, 'e016_sqpreds.parquet')
print('saved sq preds', P.shape)


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, time
F = agent_api.load_saved('allF.parquet'); tt = agent_api.train_targets()
FEAT = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = tt.merge(F.drop(columns=['future_spend_4w']), on=['household_key','snapshot_day'], how='left')
use_days = [95,123,151,179,207,235,263,291,319,347,375]
w = 0.5 ** ((431 - tr.snapshot_day)/140.0); m = tr.snapshot_day.isin(use_days)
Xw = tr.loc[m,FEAT]; yw = tr.loc[m,'future_spend_4w'].values; sw = w.loc[m].values
t0=time.time()
sq_preds = []
for seed in [7,17,27]:
    msq = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1200, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10, reg_lambda=5.0, tree_method='hist', n_jobs=8, random_state=seed)
    msq.fit(Xw, yw, sample_weight=sw)
    sq_preds.append(msq.predict(F[FEAT]))
    print('seed', seed, 'done %.0fs' % (time.time()-t0))
P = F[['household_key','snapshot_day']].copy()
for i,p in enumerate(sq_preds): P['sq%d'%i] = p
agent_api.save_table(P, 'e016_sqpreds.parquet')
print('saved sq preds', P.shape)

# ---- cell ----
import pandas as pd, numpy as np
F = agent_api.load_saved('allF.parquet'); tt = agent_api.train_targets()
S = agent_api.load_saved('e016_sqpreds.parquet')
tr = tt.merge(F.drop(columns=['future_spend_4w']), on=['household_key','snapshot_day'], how='left').merge(S, on=['household_key','snapshot_day'], how='left')
use_days = [95,123,151,179,207,235,263,291,319,347,375]
oof = tr[tr.snapshot_day.isin(use_days)]
sq = oof[['sq0','sq1','sq2']].mean(axis=1)
print('OOF MAE sq-mean %.2f  |  quantile %.2f' % ((sq-oof.future_spend_4w).abs().mean(), (oof.pred-oof.future_spend_4w).abs().mean()))
print('sq mean %.1f vs quantile mean %.1f vs act %.1f' % (sq.mean(), oof.pred.mean(), oof.future_spend_4w.mean()))
for wq in [0.0,0.25,0.5,0.75,1.0]:
    b = wq*oof.pred + (1-wq)*sq
    print('  wq=%.2f OOF MAE %.2f' % (wq, (b-oof.future_spend_4w).abs().mean()))
# geometric blend
for wq in [0.3,0.5,0.7]:
    g = np.exp(wq*np.log(np.clip(oof.pred,1e-3,None)) + (1-wq)*np.log(np.clip(sq,1e-3,None)))
    print('  geo wq=%.2f OOF MAE %.2f' % (wq, (g-oof.future_spend_4w).abs().mean()))
# held-out late-snap check
h = tr[tr.snapshot_day.isin([403,431])]
sqh = h[['sq0','sq1','sq2']].mean(axis=1)
print('held MAE: quantile %.2f sq %.2f' % ((h.pred-h.future_spend_4w).abs().mean(), (sqh-h.future_spend_4w).abs().mean()))
for wq in [0.0,0.25,0.5,0.75,1.0]:
    b = wq*h.pred + (1-wq)*sqh
    print('  held wq=%.2f MAE %.2f' % (wq, (b-h.future_spend_4w).abs().mean()))

# ---- cell ----
import pandas as pd, numpy as np
F = agent_api.load_saved('allF.parquet'); tt = agent_api.train_targets()
S = agent_api.load_saved('e016_sqpreds.parquet')
tr = tt.merge(F.drop(columns=['future_spend_4w']), on=['household_key','snapshot_day'], how='left').merge(S, on=['household_key','snapshot_day'], how='left')
use_days = [95,123,151,179,207,235,263,291,319,347,375]
oof = tr[tr.snapshot_day.isin(use_days)]
sq = oof[['sq0','sq1','sq2']].mean(axis=1)
print('OOF MAE sq-mean %.2f' % ((sq-oof.future_spend_4w).abs().mean()))
print('sq mean %.1f vs act %.1f' % (sq.mean(), oof.future_spend_4w.mean()))
for wq in [0.0,0.25,0.5,0.75,1.0]:
    b = wq*oof.future_spend_4w*0 + wq*0  # placeholder no
for wq in [0.0,0.25,0.5,0.75,1.0]:
    b = (1-wq)*sq  # need quantile preds; recompute via saved e5? quantile preds exist only for val.
print('note: quantile OOF preds not saved for train; use held-out 403/431 instead')
h = tr[tr.snapshot_day.isin([403,431])]
sqh = h[['sq0','sq1','sq2']].mean(axis=1)
print('held MAE: sq %.2f' % (sqh-h.future_spend_4w).abs().mean())
print('held sq mean %.1f act %.1f' % (sqh.mean(), h.future_spend_4w.mean()))
agent_api.save_table(h[['household_key','snapshot_day','sq0','sq1','sq2','future_spend_4w']], 'e016_held.parquet')

# ---- cell ----
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

# ---- cell ----
import pandas as pd, numpy as np
P = agent_api.load_saved('e016_allpreds.parquet')
# final E016 prediction: 0.7*quantile + 0.3*two-part (best held-out variant, 62.77)
P['prediction'] = 0.7*P.pq + 0.3*np.where(P.pc < 0.5, 0.0, P.pa)
out = P[P.snapshot_day.isin([459,487,515,543])][['household_key','snapshot_day','prediction']]
print(out.shape, sorted(out.snapshot_day.unique()))
print('mean pred %.1f (e5 was 127.1)' % out.prediction.mean())
print('finite:', np.isfinite(out.prediction).all())
agent_api.save_table(out, 'e016_preds.parquet')