
import pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
F = agent_api.load_saved('allF.parquet')
TARGET='future_spend_4w'; FEATS=[c for c in F.columns if c!=TARGET]
X=F[FEATS].astype(float); y=F[TARGET].values; days=F['snapshot_day']
tr = days<=403
# noise floor: correlations of target with trailing windows on train rows
for c in ['spend_28','spend_56','spend_84','spend_364','s1','s2','s3','s4','seq_mean']:
    print('%-10s corr=%.3f' % (c, np.corrcoef(F.loc[tr,c], y[tr])[0,1]))
# anchor MAEs on 431
idx=days==431; t=y[idx]
for c in ['spend_28','spend_84','seq_mean']:
    a=F.loc[idx,c].values
    print('anchor %-9s MAE=%.2f' % (c, np.abs(a-t).mean()))
# train p50 model
P=dict(objective='reg:quantileerror',quantile_alpha=0.5,max_depth=6,min_child_weight=10,subsample=0.8,
       colsample_bytree=0.7,tree_method='hist',eval_metric=['mae'],learning_rate=0.03,seed=0)
m=days<=403; w=0.5**((403-days[m]).astype(float)/140)
bst=xgb.train(P,xgb.DMatrix(X[m],label=y[m],weight=w),1200,verbose_eval=False)
p=bst.predict(xgb.DMatrix(X[idx]))
print('model MAE=%.3f' % np.abs(p-t).mean())
s28=F.loc[idx,'spend_28'].values; s84=F.loc[idx,'spend_84'].values
for lam in [0.05,0.1,0.2,0.3,0.5]:
    for name,a in [('s28',s28),('s84',s84)]:
        q=p+lam*(a-p)
        print('blend lam=%.2f %-4s: MAE=%.3f' % (lam,name,np.abs(q-t).mean()))
