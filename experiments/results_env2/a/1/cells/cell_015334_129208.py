
import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings('ignore')
F = agent_api.load_saved('allF.parquet')
TARGET='future_spend_4w'; FEATS=[c for c in F.columns if c!=TARGET]
X=F[FEATS].astype(float); y=F[TARGET].values; days=F['snapshot_day']
BASE=dict(objective='reg:quantileerror',quantile_alpha=0.5,max_depth=6,min_child_weight=10,subsample=0.8,
          colsample_bytree=0.7,tree_method='hist',eval_metric=['mae'],learning_rate=0.03,seed=0)

def ev(cfg, tr_max, evd, rounds=1200, half=140):
    P=dict(BASE); P.update(cfg)
    m=days<=tr_max; w=0.5**((tr_max-days[m]).astype(float)/half)
    bst=xgb.train(P, xgb.DMatrix(X[m],label=y[m],weight=w), rounds, verbose_eval=False)
    idx=days==evd; p=bst.predict(xgb.DMatrix(X[idx]))
    return np.abs(p-y[idx]).mean()

t0=time.time()
print('ref (tr403,ev431): %.3f' % ev({},403,431))
print('tr375,ev431      : %.3f  <-- training-window effect' % ev({},375,431))
print('tr403,ev403? n/a')
print('mcw=5 : %.3f' % ev({'min_child_weight':5},403,431))
print('mcw=20: %.3f' % ev({'min_child_weight':20},403,431))
print('depth5: %.3f' % ev({'max_depth':5},403,431))
print('depth7: %.3f' % ev({'max_depth':7},403,431))
print('half100: %.3f' % ev({},403,431,half=100))
print('half200: %.3f' % ev({},403,431,half=200))
print('l1=1  : %.3f' % ev({'alpha':1.0},403,431))
print('l1=5  : %.3f' % ev({'alpha':5.0},403,431))
print('maxbin=256: %.3f' % ev({'max_bin':256},403,431))
print('cs=0.8: %.3f' % ev({'colsample_bytree':0.8},403,431))
print('ss=0.9: %.3f' % ev({'subsample':0.9},403,431))
print('tweedie1.3: %.3f' % ev({'objective':'reg:tweedie','tweedie_variance_power':1.3},403,431))
print('tweedie1.6: %.3f' % ev({'objective':'reg:tweedie','tweedie_variance_power':1.6},403,431))
print('%.0fs'%(time.time()-t0))
