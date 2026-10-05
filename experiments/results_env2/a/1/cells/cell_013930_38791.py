
import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings('ignore')
F = agent_api.load_saved('allF.parquet')
TARGET='future_spend_4w'; FEATS=[c for c in F.columns if c!=TARGET]
X=F[FEATS].astype(float); y=F[TARGET].values; days=F['snapshot_day']

def train_eval(params, tr_max=403, ev=431, rounds=2400, half=140, also_insample=False, lr=0.03):
    P=dict(params); P['learning_rate']=lr
    m=days<=tr_max
    w=0.5**((tr_max-days[m]).astype(float)/half)
    bst=xgb.train(P, xgb.DMatrix(X[m],label=y[m],weight=w), rounds, verbose_eval=False)
    idx=days==ev; p=bst.predict(xgb.DMatrix(X[idx])); t=y[idx]
    mae=np.abs(p-t).mean()
    line='MAE=%.3f bias=%+.2f' % (mae,(p-t).mean())
    if also_insample:
        pi=bst.predict(xgb.DMatrix(X[m])); ti=y[m]
        line+='  | in-sample bias=%+.2f' % (pi-ti).mean()
        hi=ti>=np.quantile(ti,0.9)
        line+=' insample-top10 MAE=%.1f (pred %.0f vs true %.0f)' % (np.abs(pi[hi]-ti[hi]).mean(), pi[hi].mean(), ti[hi].mean())
    hi=t>=np.quantile(t,0.9)
    line+='  | top10: pred %.0f true %.0f MAE %.1f' % (p[hi].mean(), t[hi].mean(), np.abs(p[hi]-t[hi]).mean())
    print(line)
    return mae, bst

P50=dict(objective='reg:quantileerror',quantile_alpha=0.5,max_depth=6,min_child_weight=10,subsample=0.8,colsample_bytree=0.7,tree_method='hist',eval_metric=['mae'])
t0=time.time()
print('ref a=.5 (insample check):'); train_eval(P50, also_insample=True)
print('a=.55:'); train_eval({**P50,'quantile_alpha':0.55})
print('a=.60:'); train_eval({**P50,'quantile_alpha':0.60})
print('depth8 mcw5:'); train_eval({**P50,'max_depth':8,'min_child_weight':5})
print('%.0fs' % (time.time()-t0))
