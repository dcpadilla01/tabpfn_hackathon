
import pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
F = agent_api.load_saved('allF.parquet')
TARGET='future_spend_4w'; FEATS=[c for c in F.columns if c!=TARGET]
P = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=6, min_child_weight=10,
         subsample=0.8, colsample_bytree=0.7, tree_method='hist', learning_rate=0.03, eval_metric=['mae'])
days = F['snapshot_day']; X=F[FEATS].astype(float); y=F[TARGET].values
m = days<=403
w = 0.5**((403-days[m]).astype(float)/140)
bst = xgb.train(P, xgb.DMatrix(X[m],label=y[m],weight=w), 2400, verbose_eval=False)
idx = days==431
p = bst.predict(xgb.DMatrix(X[idx])); t = y[idx]
print('overall MAE %.3f  bias %+.3f' % (np.abs(p-t).mean(), (p-t).mean()))
bins=[0,1,25,75,150,300,600,1e9]
for lo,hi in zip(bins[:-1],bins[1:]):
    sel=(t>=lo)&(t<hi)
    if sel.sum()==0: continue
    print('t in [%6.0f,%6.0f): n=%4d  pred_mean=%7.1f  truth_mean=%7.1f  MAE=%7.1f  share_of_total_loss=%.2f' %
          (lo,hi,sel.sum(),p[sel].mean(),t[sel].mean(),np.abs(p[sel]-t[sel]).mean(),np.abs(p[sel]-t[sel]).sum()/np.abs(p-t).sum()))
# zero rows
sel = t==0
print('zero rows: n=%d pred mean %.1f  MAE contribution %.2f' % (sel.sum(), p[sel].mean(), np.abs(p[sel]-t[sel]).sum()/np.abs(p-t).sum()))
# top decile
q=np.quantile(t,0.9); sel=t>=q
print('top-decile rows: MAE=%.1f share=%.2f  (n=%d)' % (np.abs(p[sel]-t[sel]).mean(), np.abs(p[sel]-t[sel]).sum()/np.abs(p-t).sum(), sel.sum()))
