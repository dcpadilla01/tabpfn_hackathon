import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

t = agent_api.load_saved('e008_fwd_calendar.parquet')
tt = agent_api.train_targets()
days = agent_api.snapshot_days()
tr_days, va_days = days['train'], days['validation']
print('train days', tr_days); print('val days', va_days)

df = tt.merge(t, on=['household_key','snapshot_day'], how='inner')
feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
tr = df[df.snapshot_day.isin(tr_days)].reset_index(drop=True)
va = df[df.snapshot_day.isin(va_days)].reset_index(drop=True)
print('train rows', len(tr), 'val rows', len(va))

ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values

def make_X(d, impute):
    X = d[feats].astype(float).copy()
    if impute == 'zero':
        X = X.fillna(0.0)
    else:  # train medians
        med = tr[feats].median()
        X = X.fillna(med)
    return X.values

Xtr_raw = make_X(tr,'zero'); Xva_raw = make_X(va,'zero')
Xtr_med = make_X(tr,'med'); Xva_med = make_X(va,'med')

def fit_eval(Xtr, ytr, Xva, yva, alpha, standardize):
    if standardize:
        mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
        Xtr2=(Xtr-mu)/sd; Xva2=(Xva-mu)/sd
    else:
        Xtr2, Xva2 = Xtr, Xva
    Xtr1 = np.hstack([Xtr2, np.ones((len(Xtr2),1))]); Xva1 = np.hstack([Xva2, np.ones((len(Xva2),1))])
    A = Xtr1.T@Xtr1 + alpha*np.eye(Xtr1.shape[1]); A[-1,-1]-=alpha
    w = np.linalg.solve(A, Xtr1.T@ytr)
    p = Xva1@w
    mae = np.abs(p-yva).mean()
    r2 = 1-((p-yva)**2).sum()/((yva-yva.mean())**2).sum()
    return mae, r2

print('\nconfig search (target val MAE ~61.109):')
for name,(Xa,Xb) in {'zero':(Xtr_raw,Xva_raw),'med':(Xtr_med,Xva_med)}.items():
    for std in [False,True]:
        for alpha in [0.0,1.0,10.0,100.0,1000.0]:
            mae,r2 = fit_eval(Xa,ytr,Xb,yva,alpha,std)
            print(f'impute={name:4s} std={std!s:5s} alpha={alpha:7.1f}  valMAE={mae:7.3f}  R2={r2:.4f}')