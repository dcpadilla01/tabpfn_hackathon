
import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings('ignore')
F = agent_api.load_saved('allF.parquet')
TARGET='future_spend_4w'; FEATS=[c for c in F.columns if c!=TARGET]
X=F[FEATS].astype(float); y=F[TARGET].values; days=F['snapshot_day']
P50=dict(objective='reg:quantileerror',quantile_alpha=0.5,max_depth=6,min_child_weight=10,subsample=0.8,colsample_bytree=0.7,tree_method='hist',learning_rate=0.03,eval_metric=['mae'])

def get_preds(alpha, tr_max=403, ev=431, rounds=2400, half=140, seed=0):
    P=dict(P50); P['quantile_alpha']=alpha; P['seed']=seed
    m=days<=tr_max
    w=0.5**((tr_max-days[m]).astype(float)/half)
    bst=xgb.train(P, xgb.DMatrix(X[m],label=y[m],weight=w), rounds, verbose_eval=False)
    return bst.predict(xgb.DMatrix(X[days==ev])), y[days==ev]

t0=time.time()
p50,t = get_preds(0.5); p60,_ = get_preds(0.6); p70,_ = get_preds(0.7)
def ev_mae(p, t): return np.abs(p-t).mean()
print('p50 %.3f | p60 %.3f | p70 %.3f' % (ev_mae(p50,t), ev_mae(p60,t), ev_mae(p70,t)))
# conditional blend: s = sigmoid of p50 around threshold
for thr in [100,150,200,300]:
    for wgt in [0.3,0.5,0.8,1.0]:
        s = np.clip((p50-thr)/thr, 0, 1)*wgt   # 0 below thr, ->wgt far above
        p = p50 + s*(p60-p50)
        print('thr=%3d w=%.1f: MAE=%.3f (top10 %.1f)' % (thr,wgt,ev_mae(p,t),np.abs(p[t>=np.quantile(t,0.9)]-t[t>=np.quantile(t,0.9)]).mean()))
# simple global blends for reference
for a in [0.2,0.3]:
    print('global p50+(%.1f)(p60-p50): %.3f' % (a, ev_mae(p50+a*(p60-p50),t)))
print('%.0fs'%(time.time()-t0))
