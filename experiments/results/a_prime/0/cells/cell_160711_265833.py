
import agent_api as A, pandas as pd, numpy as np
e013 = A.load_saved('e013_stock.parquet')
tt = A.train_targets()
df = e013.merge(tt, on=['household_key','snapshot_day'])
tr = df[df.snapshot_day<=403]; va = df[df.snapshot_day==431]
feats = [c for c in e013.columns if c not in ('household_key','snapshot_day')]
def prep(d):
    X = d[feats].copy()
    for c in X.columns:
        if X[c].dtype==bool: X[c]=X[c].astype(float)
    # simple numeric encoding for categoricals
    X = pd.get_dummies(X.astype(object).where(~X.columns.str.startswith(('classification','homeowner','kid')), X), dummy_na=True) if False else X
    return X
# quick: baseline predictors on 431
yv = va.future_spend_4w.values
print('pred global median  MAE', np.abs(yv-np.median(tr.future_spend_4w)).mean())
print('pred zero           MAE', np.abs(yv-0).mean())
# household last block p1 = spend_28
print('pred spend_28       MAE', np.abs(yv-va.spend_28.values).mean())
print('pred 0.8*spend_28   MAE', np.abs(yv-0.8*va.spend_28.values).mean())
print('pred max(0,spend_28-10) MAE', np.abs(yv-np.maximum(0,va.spend_28.values-10)).mean())
# error decomposition for spend_28 predictor
err = yv-va.spend_28.values
print('by yv quantile:')
q = pd.qcut(yv, 5, duplicates='drop')
print(pd.DataFrame({'y':yv,'p':va.spend_28.values,'e':err}).groupby(q,observed=True).apply(lambda g: pd.Series({'n':len(g),'y_mean':g.y.mean(),'p_mean':g.p.mean(),'mae':np.abs(g.e).mean()})))
print('zero-y rows:', (yv==0).sum(), 'MAE on them', np.abs(err[yv==0]).mean(), 'mean pred on them', va.spend_28.values[yv==0].mean())
