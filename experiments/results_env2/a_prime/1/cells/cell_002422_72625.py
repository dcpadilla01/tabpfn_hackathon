
import pandas as pd, numpy as np, time

t15 = agent_api.load_saved('e015_stack.parquet')
tt = agent_api.train_targets()
drop = {'household_key','snapshot_day','stack_ridge_log','stack_ridge','index'}
feat_cols = [c for c in t15.columns if c not in drop]
cat_cols = ['classification_1','classification_2','classification_3','classification_4',
            'classification_5','homeowner_desc','kid_category_desc']
num_cols = [c for c in feat_cols if c not in cat_cols]
df = t15.merge(tt, on=['household_key','snapshot_day'], how='left')
train_mask = df['future_spend_4w'].notna().values
Xn = df[num_cols].astype(float); Xn = Xn.fillna(Xn.median())
Xc = pd.get_dummies(df[cat_cols].astype('category'), dummy_na=True, dtype=float)
X = np.hstack([Xn.values, Xc.values])
y = df['future_spend_4w'].values; ly = np.log1p(y)
sd = df['snapshot_day'].values
train_days = sorted(pd.unique(sd[train_mask]))

def ridge_fit_pred(Xtr, ytr, Xte, alpha):
    mu, sg = Xtr.mean(0), Xtr.std(0)+1e-9
    A = (Xtr-mu)/sg; B = (Xte-mu)/sg
    A = np.hstack([A, np.ones((len(A),1))]); B = np.hstack([B, np.ones((len(B),1))])
    P = np.eye(A.shape[1]); P[-1,-1]=0.0
    return B @ np.linalg.solve(A.T@A + alpha*P, A.T@ytr)

# per-snapshot bias of log ridge on train (LOSO)
oof_log = np.zeros(len(df))
for s in train_days:
    m = (sd==s) & train_mask; mfit = train_mask & (sd!=s)
    oof_log[m] = ridge_fit_pred(X[mfit], ly[mfit], X[m], 300.0)
pred_log = np.expm1(oof_log)
resid = y[train_mask] - pred_log[train_mask]
print("per-snapshot mean resid (train LOSO, log ridge):")
for s in train_days:
    m = (sd==s) & train_mask
    print(f"  day {s}: n={m.sum()}, mean_resid={resid[sd[m]].mean():8.2f}, mean_y={y[m].mean():8.2f}, bias%={100*resid[sd[m]].mean()/max(y[m].mean(),1e-9):6.1f}")

# blend check: E015's own stack vs raw log ridge on val — can't see val, but check corr of stack cols
print("\nstack_ridge vs stack_ridge_log corr:", np.corrcoef(t15['stack_ridge'], t15['stack_ridge_log'])[0,1].round(3))
