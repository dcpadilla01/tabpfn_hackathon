
import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e005_decay_gapcv.parquet')
tt = agent_api.train_targets()
m = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
train_days = agent_api.snapshot_days()['train']

FEATS = [c for c in t.columns if c not in ('household_key','snapshot_day')]

def ridge_fit_pred(Xtr, ytr, Xva, lam=1.0):
    # standardize
    mu = np.nanmean(Xtr, axis=0); sd = np.nanstd(Xtr, axis=0); sd[sd==0]=1
    Ztr = (Xtr-mu)/sd; Zva = (Xva-mu)/sd
    Ztr = np.nan_to_num(Ztr, nan=0.0); Zva = np.nan_to_num(Zva, nan=0.0)
    # add intercept + snapshot one-hot handled outside
    A = Ztr.T@Ztr + lam*np.eye(Ztr.shape[1])
    b = Ztr.T@ytr
    w = np.linalg.solve(A, b)
    return Zva@w

def cv_mae(df, feats, target='future_spend_4w', log_target=False, lam=1.0, add_day_dummies=True):
    df = df.copy()
    y = df[target].values.astype(float)
    yt = np.log1p(y) if log_target else y
    maes=[]
    for d in train_days:
        tr = df['snapshot_day']!=d; va = df['snapshot_day']==d
        Xtr = df.loc[tr, feats].values.astype(float); Xva = df.loc[va, feats].values.astype(float)
        if add_day_dummies:
            D = pd.get_dummies(df['snapshot_day']).astype(float)
            Dtr = D.loc[tr].values; Dva = D.loc[va].values
            Xtr = np.hstack([Xtr, Dtr]); Xva = np.hstack([Xva, Dva])
        p = ridge_fit_pred(Xtr, yt[tr.values], Xva, lam)
        if log_target: p = np.expm1(p)
        p = np.clip(p, 0, None)
        maes.append(np.mean(np.abs(p - y[va.values])))
    return float(np.mean(maes))

base = cv_mae(m, FEATS)
base_log = cv_mae(m, FEATS, log_target=True)
print('E005 ridge proxy MAE raw-target:', round(base,3))
print('E005 ridge proxy MAE log-target:', round(base_log,3))
# simple baselines
for name, pred in [('global median', np.full(len(m), m['future_spend_4w'].median())),
                   ('ew28*1.0', m['ew_28'].values), ('spend28*1.0', m['spend_28'].values)]:
    print(name, round(float(np.mean(np.abs(pred - m['future_spend_4w']))),3))
