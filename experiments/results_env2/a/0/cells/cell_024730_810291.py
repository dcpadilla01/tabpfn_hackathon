
import agent_api as A, numpy as np, pandas as pd

feats = A.load_saved('feats_v4.parquet')
tt = A.train_targets()
tr = feats.merge(tt, on=['household_key','snapshot_day'])
print('train rows', len(tr))
y = tr.future_spend_4w.values
print('y: mean %.1f median %.1f p90 %.1f p99 %.1f zero-share %.3f' % (y.mean(), np.median(y), np.percentile(y,90), np.percentile(y,99), (y==0).mean()))

# naive predictors
def mae(p): return np.abs(np.asarray(p)-y).mean()
print('const median %.2f | const mean %.2f | zero %.2f' % (mae(np.median(y)), mae(y.mean()), mae(np.zeros_like(y))))
for c in ['exp4w_blend','lag1_spend','spend_28','exp4w_all','exp4w_84']:
    print(c, 'MAE %.2f' % mae(tr[c].values))
oof = A.load_saved('oof_e013.parquet')
tr2 = tr.merge(oof[['household_key','snapshot_day','oof']], on=['household_key','snapshot_day'])
print('oof MAE %.3f' % mae(tr2.oof.values))
for w in [0.1,0.2,0.3]:
    print('oof*(1-w)+exp4w_blend*w, w=%.1f MAE %.3f' % (w, mae((1-w)*tr2.oof.values + w*tr2.exp4w_blend.values)))
# residual stats
res = y - tr.exp4w_blend.values
print('residual: mean %.1f median %.1f std %.1f' % (res.mean(), np.median(res), res.std()))
print('corr(oof, y) %.3f | corr(oof, exp4w_blend) %.3f' % (np.corrcoef(tr2.oof, tr2.future_spend_4w)[0,1], np.corrcoef(tr2.oof, tr2.exp4w_blend)[0,1]))

# saved val preds: correlations
ps = {n: A.load_saved('pred_e%03d.parquet'%n) for n in [4,5,7,11,13,19]}
base = ps[13][['household_key','snapshot_day']].copy()
m = base.copy()
for n,p in ps.items():
    m = m.merge(p.rename(columns={'prediction':'p%d'%n}), on=['household_key','snapshot_day'])
print(m.shape)
print(m[[c for c in m.columns if c.startswith('p')]].corr().round(3))
for n in [4,5,7,11,19]:
    print('blend e013+e%03d 50/50 MAE:'%n, end=' ')
    mm = m.merge(tt, on=['household_key','snapshot_day']) if False else m
    # no targets for val; just corr shown above
print()
import time
t0=time.time()
import xgboost as xgb
X = tr[[c for c in feats.columns if c not in ('household_key','snapshot_day')]].head(20000)
dtr = xgb.DMatrix(X, label=y[:20000])
b = xgb.train({'objective':'reg:quantileerror','quantile_alpha':0.5,'max_depth':5,'min_child_weight':40,'learning_rate':0.08,'nthread':4}, dtr, 400)
print('one xgb fit (20k rows, 400 trees): %.1fs' % (time.time()-t0))
