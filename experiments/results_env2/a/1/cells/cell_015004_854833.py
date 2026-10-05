
import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings('ignore')
F = agent_api.load_saved('allF.parquet')
TARGET='future_spend_4w'; FEATS=[c for c in F.columns if c!=TARGET]
X=F[FEATS].astype(float); y=F[TARGET].values; days=F['snapshot_day']
BASE=dict(objective='reg:quantileerror',quantile_alpha=0.5,max_depth=6,min_child_weight=10,subsample=0.8,
          colsample_bytree=0.7,tree_method='hist',eval_metric=['mae'],learning_rate=0.03)
def one_model(P, tr_max, ev, rounds=1200, half=140):
    m=days<=tr_max; w=0.5**((tr_max-days[m]).astype(float)/half)
    bst=xgb.train(P, xgb.DMatrix(X[m],label=y[m],weight=w), rounds, verbose_eval=False)
    return bst.predict(xgb.DMatrix(X[days==ev]))
t0=time.time()
p1 = one_model(dict(BASE,seed=0),403,431); t=y[days==431]
print('1200r single: MAE=%.3f' % np.abs(p1-t).mean())
P8=[one_model(dict(BASE,seed=s),403,431) for s in range(8)]
p8=np.mean(P8,axis=0)
print('1200r 8-seed: MAE=%.3f  member MAEs: %s' % (np.abs(p8-t).mean(), ' '.join('%.2f'%np.abs(q-t).mean() for q in P8)))
print('%.0fs'%(time.time()-t0))
