
import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings('ignore')
F = agent_api.load_saved('allF.parquet')
TARGET='future_spend_4w'; FEATS=[c for c in F.columns if c!=TARGET]
X=F[FEATS].astype(float); y=F[TARGET].values; days=F['snapshot_day']
BASE=dict(objective='reg:quantileerror',quantile_alpha=0.5,max_depth=6,min_child_weight=10,subsample=0.8,
          colsample_bytree=0.7,tree_method='hist',eval_metric=['mae'],learning_rate=0.03)

def one_model(P, tr_max, ev, rounds=2400, half=140):
    m=days<=tr_max; w=0.5**((tr_max-days[m]).astype(float)/half)
    bst=xgb.train(P, xgb.DMatrix(X[m],label=y[m],weight=w), rounds, verbose_eval=False)
    return bst.predict(xgb.DMatrix(X[days==ev]))

t0=time.time()
# single seed
p1 = one_model(dict(BASE,seed=0), 403, 431); t=y[days==431]
print('single seed: MAE=%.3f' % np.abs(p1-t).mean())
# 8-seed ensemble, same config
P8=[]
for s in range(8):
    P8.append(one_model(dict(BASE,seed=s), 403, 431))
p8=np.mean(P8,axis=0)
print('8-seed: MAE=%.3f (spread of members: %.3f-%.3f)' % (np.abs(p8-t).mean(), min(np.abs(q-t).mean() for q in P8), max(np.abs(q-t).mean() for q in P8)))
# 4-seed x 2 configs (subsample/colsample jitter)
P16=[]
for s in range(4):
    for (ss,cs,md,mcw) in [(0.8,0.7,6,10),(0.7,0.6,6,8)]:
        P16.append(one_model(dict(BASE,seed=s,subsample=ss,colsample_bytree=cs,max_depth=md,min_child_weight=mcw),403,431))
p16=np.mean(P16,axis=0)
print('8-mixed: MAE=%.3f' % np.abs(p16-t).mean())
print('%.0fs'%(time.time()-t0))
