
import pandas as pd, numpy as np

t15 = agent_api.load_saved('e015_stack.parquet')
print("E015 shape:", t15.shape)
print("E015 cols:", t15.columns.tolist())
print("\nrows with any NaN per snapshot_day (E015):")
print(t15.groupby('snapshot_day').apply(lambda g: int(g.isna().any(axis=1).sum())))

t12 = agent_api.load_saved('e012_style.parquet')
print("\nE012 shape:", t12.shape)
print("rows ALL-NaN per snapshot_day (E012):")
print(t12.groupby('snapshot_day').apply(lambda g: int(g.isna().all(axis=1).sum())))

tt = agent_api.train_targets()
print("\ntrain_targets:", tt.shape, tt.columns.tolist())
print(tt.head(3))
print("target describe:\n", tt[agent_api.TARGET].describe())


# ---- cell ----

import pandas as pd, numpy as np

rr = agent_api.load_saved('rawrec.parquet')
print("rawrec shape:", rr.shape)
print("cols:", rr.columns.tolist())
print("\nrows per snapshot_day:")
print(rr.groupby('snapshot_day').size())
print("\nNaN counts per column (sample):")
print(rr.isna().sum().sort_values(ascending=False).head(20))
print("\nhead:\n", rr.head(3))
print("\nall-NaN rows per snapshot_day:")
print(rr.groupby('snapshot_day').apply(lambda g: int(g.isna().all(axis=1).sum())))


# ---- cell ----

import pandas as pd, numpy as np
rr = agent_api.load_saved('rawrec.parquet')
print(rr[['g','sp28','sp56','ew','wk28']].head(8))
print("\nunique g:", rr['g'].nunique(), " rows:", len(rr))
# check if g encodes household|day
sample = rr['g'].astype(str).head(5).tolist()
print("g samples:", sample)
# how many rows per household?
rr['hh'] = rr['g'].astype(str).str.split('|').str[0]
print("\nrows per hh (describe):", rr.groupby('hh').size().describe())
# NaN structure
print("\nNaN cols:\n", rr.isna().sum().sort_values(ascending=False).head(8))
print("\nany-nan rows:", int(rr.isna().any(axis=1).sum()))


# ---- cell ----

import pandas as pd, numpy as np, time

t15 = agent_api.load_saved('e015_stack.parquet')
tt = agent_api.train_targets()

drop = {'household_key','snapshot_day','stack_ridge_log','stack_ridge','index'}
feat_cols = [c for c in t15.columns if c not in drop]
cat_cols = ['classification_1','classification_2','classification_3','classification_4',
            'classification_5','homeowner_desc','kid_category_desc']
num_cols = [c for c in feat_cols if c not in cat_cols]

df = t15.merge(tt, on=['household_key','snapshot_day'], how='left')
train_mask = df['future_spend_4w'].notna()

Xn = df[num_cols].astype(float)
Xn = Xn.fillna(Xn.median())
Xc = pd.get_dummies(df[cat_cols].astype('category'), dummy_na=True, dtype=float)
X = np.hstack([Xn.values, Xc.values])
y = df['future_spend_4w'].values
ly = np.log1p(y)
sd = df['snapshot_day'].values
train_days = sorted(df.loc[train_mask,'snapshot_day'].unique())

def ridge_fit_pred(Xtr, ytr, Xte, alpha):
    mu, sg = Xtr.mean(0), Xtr.std(0)+1e-9
    A = (Xtr-mu)/sg; B = (Xte-mu)/sg
    A = np.hstack([A, np.ones((len(A),1))]); B = np.hstack([B, np.ones((len(B),1))])
    d = A.shape[1]
    P = np.eye(d); P[-1,-1] = 0.0
    w = np.linalg.solve(A.T@A + alpha*P, A.T@ytr)
    return B@w

t0=time.time()
oof = np.zeros(train_mask.sum())
for s in train_days:
    m = (sd==s) & train_mask
    mfit = train_mask & (sd!=s)
    oof[m] = ridge_fit_pred(X[mfit], ly[mfit], X[m], 300.0)
mae_log = np.mean(np.abs(np.expm1(oof) - y[train_mask]))
print("LOSO ridge log alpha=300  LOSO-MAE(train):", round(mae_log,3), " time:", round(time.time()-t0,1))


# ---- cell ----

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

Xn = df[num_cols].astype(float)
Xn = Xn.fillna(Xn.median())
Xc = pd.get_dummies(df[cat_cols].astype('category'), dummy_na=True, dtype=float)
X = np.hstack([Xn.values, Xc.values])
y = df['future_spend_4w'].values
ly = np.log1p(y)
sd = df['snapshot_day'].values
train_days = sorted(pd.unique(sd[train_mask]))

def ridge_fit_pred(Xtr, ytr, Xte, alpha):
    mu, sg = Xtr.mean(0), Xtr.std(0)+1e-9
    A = (Xtr-mu)/sg; B = (Xte-mu)/sg
    A = np.hstack([A, np.ones((len(A),1))]); B = np.hstack([B, np.ones((len(B),1))])
    d = A.shape[1]
    P = np.eye(d); P[-1,-1] = 0.0
    w = np.linalg.solve(A.T@A + alpha*P, A.T@ytr)
    return B@w

t0=time.time()
oof = np.zeros(train_mask.sum())
for s in train_days:
    m = (sd==s) & train_mask
    mfit = train_mask & (sd!=s)
    oof[m] = ridge_fit_pred(X[mfit], ly[mfit], X[m], 300.0)
mae_log = np.mean(np.abs(np.expm1(oof) - y[train_mask]))
print("LOSO ridge log alpha=300  LOSO-MAE(train):", round(mae_log,3), " time:", round(time.time()-t0,1))


# ---- cell ----

import pandas as pd, numpy as np
t15 = agent_api.load_saved('e015_stack.parquet')
tt = agent_api.train_targets()
print("t15:", t15.shape, "tt:", tt.shape)
print("t15 dup keys:", t15.duplicated(['household_key','snapshot_day']).sum())
print("tt dup keys:", tt.duplicated(['household_key','snapshot_day']).sum())
print("t15 dtypes hh/sd:", t15['household_key'].dtype, t15['snapshot_day'].dtype)
print("tt dtypes hh/sd:", tt['household_key'].dtype, tt['snapshot_day'].dtype)
df = t15.merge(tt, on=['household_key','snapshot_day'], how='left')
print("merged:", df.shape)
print("t15 index type:", t15.index.dtype, "tt index:", tt.index.dtype)


# ---- cell ----

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

Xn = df[num_cols].astype(float)
Xn = Xn.fillna(Xn.median())
Xc = pd.get_dummies(df[cat_cols].astype('category'), dummy_na=True, dtype=float)
X = np.hstack([Xn.values, Xc.values])
y = df['future_spend_4w'].values
ly = np.log1p(y)
sd = df['snapshot_day'].values
train_days = sorted(pd.unique(sd[train_mask]))

def ridge_fit_pred(Xtr, ytr, Xte, alpha):
    mu, sg = Xtr.mean(0), Xtr.std(0)+1e-9
    A = (Xtr-mu)/sg; B = (Xte-mu)/sg
    A = np.hstack([A, np.ones((len(A),1))]); B = np.hstack([B, np.ones((len(B),1))])
    d = A.shape[1]
    P = np.eye(d); P[-1,-1] = 0.0
    w = np.linalg.solve(A.T@A + alpha*P, A.T@ytr)
    return B@w

t0=time.time()
oof = np.zeros(len(df))
for s in train_days:
    m = (sd==s) & train_mask
    mfit = train_mask & (sd!=s)
    oof[m] = ridge_fit_pred(X[mfit], ly[mfit], X[m], 300.0)
mae_log = np.mean(np.abs(np.expm1(oof[train_mask]) - y[train_mask]))
print("LOSO ridge(log,alpha=300) LOSO-MAE(train):", round(mae_log,3), "time:", round(time.time()-t0,1))

# also linear target for comparison
oof2 = np.zeros(len(df))
for s in train_days:
    m = (sd==s) & train_mask
    mfit = train_mask & (sd!=s)
    oof2[m] = ridge_fit_pred(X[mfit], y[mfit], X[m], 300.0)
mae_lin = np.mean(np.abs(oof2[train_mask] - y[train_mask]))
print("LOSO ridge(lin,alpha=300) LOSO-MAE(train):", round(mae_lin,3))


# ---- cell ----

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
