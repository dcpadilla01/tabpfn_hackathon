
import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings('ignore')
F = agent_api.load_saved('allF.parquet')
TARGET='future_spend_4w'; FEATS=[c for c in F.columns if c!=TARGET]
X=F[FEATS].astype(float); y=F[TARGET].values; days=F['snapshot_day']
BASE=dict(objective='reg:quantileerror',quantile_alpha=0.5,max_depth=6,min_child_weight=10,subsample=0.8,
          colsample_bytree=0.7,tree_method='hist',eval_metric=['mae'],learning_rate=0.03)
t0=time.time()
m=days<=403
w=0.5**((403-days[m]).astype(float)/140)
dtr=xgb.DMatrix(X[m],label=y[m],weight=w)
seeds=[]
for s in range(8):
    bst=xgb.train(dict(BASE,seed=s), dtr, 1200, verbose_eval=False)
    seeds.append(bst.predict(xgb.DMatrix(X)))
print('8 models trained (%.0fs)'%(time.time()-t0))
p50 = np.mean(seeds,axis=0)
val = days>=459
s28 = F.loc[val,'spend_28'].values
p_final = 0.9*p50[val] + 0.1*s28
sub = F.loc[val,['household_key','snapshot_day']].copy()
sub['prediction']=p_final
print('val rows:',len(sub),'days:',sorted(sub.snapshot_day.unique()),'nan:',sub.prediction.isna().sum())
print('pred stats: mean %.1f med %.1f p90 %.1f max %.1f' % (p_final.mean(),np.median(p_final),np.quantile(p_final,0.9),p_final.max()))
e5 = agent_api.load_saved('e005_preds.parquet').set_index(['household_key','snapshot_day'])['prediction']
mm = sub.set_index(['household_key','snapshot_day']).join(e5.rename('e5'))
print('vs E005: corr %.4f  mean|diff| %.2f  mean(diff) %+.2f' % (np.corrcoef(mm.prediction,mm.e5)[0,1],(mm.prediction-mm.e5).abs().mean(),(mm.prediction-mm.e5).mean()))
path = agent_api.save_table(sub, 'e011_preds.parquet')
print('saved:', path)
