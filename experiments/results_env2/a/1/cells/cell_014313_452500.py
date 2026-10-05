
import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings('ignore')
F = agent_api.load_saved('allF.parquet')
TARGET='future_spend_4w'; FEATS=[c for c in F.columns if c!=TARGET]
X=F[FEATS].astype(float); y=F[TARGET].values; days=F['snapshot_day']
BASE=dict(max_depth=6,min_child_weight=10,subsample=0.8,colsample_bytree=0.7,tree_method='hist',eval_metric=['mae'],learning_rate=0.03)

def mkp(obj, alpha=None, seed=0):
    P=dict(BASE); P['objective']=obj; P['seed']=seed
    if alpha is not None: P['quantile_alpha']=alpha
    return P

def preds(P, tr_max=403, ev=431, rounds=2400, half=140):
    m=days<=tr_max; w=0.5**((tr_max-days[m]).astype(float)/half)
    bst=xgb.train(P, xgb.DMatrix(X[m],label=y[m],weight=w), rounds, verbose_eval=False)
    return bst.predict(xgb.DMatrix(X[days==ev])), y[days==ev]

t0=time.time()
p50,t = preds(mkp('reg:quantileerror',0.5))
pm,_  = preds(mkp('reg:squarederror'))
print('p50 %.3f | mean-model %.3f | corr %.4f' % (np.abs(p50-t).mean(), np.abs(pm-t).mean(), np.corrcoef(p50,pm)[0,1]))
for lam in [0.1,0.2,0.3,0.4,0.5,0.7]:
    p = p50+lam*(pm-p50)
    hi=t>=np.quantile(t,0.9)
    print('lam=%.1f: MAE=%.3f top10=%.1f' % (lam, np.abs(p-t).mean(), np.abs(p[hi]-t[hi]).mean()))
# two-stage residual model
m=days<=403; w=0.5**((403-days[m]).astype(float)/140)
dtr=xgb.DMatrix(X[m],label=y[m],weight=w); bst50=xgb.train(mkp('reg:quantileerror',0.5),dtr,2400,verbose_eval=False)
resid = y[m]-bst50.predict(dtr)
dres=xgb.DMatrix(X[m],label=resid,weight=w)
for rr in [600,1200]:
    bstr=xgb.train(mkp('reg:squarederror'),dres,rr,verbose_eval=False)
    p2 = p50 + bstr.predict(xgb.DMatrix(X[days==431]))
    hi=t>=np.quantile(t,0.9)
    print('resid2stage rounds=%d: MAE=%.3f top10=%.1f' % (rr,np.abs(p2-t).mean(),np.abs(p[hi]-t[hi]).mean()))
print('%.0fs'%(time.time()-t0))
