
import numpy as np, pandas as pd

tt = agent_api.train_targets()
y = tt.future_spend_4w.values

def mae(pred):
    return round(np.mean(np.abs(y - pred)),3)

season = agent_api.load_saved('season.parquet')
m = tt.merge(season, on=['household_key','snapshot_day'], how='left')

# raw predictors
print('own_ly_spend4w MAE:', mae(m.own_ly_spend4w.values))
print('own_ly2_spend4w MAE:', mae(m.own_ly2_spend4w.values))
print('season_lift MAE:', mae(m.season_lift.values))
print('spend56 MAE:', mae(m.spend56.values))
print('spend112 MAE:', mae(m.spend112.values))
print('blend .5*28+.5*56:', mae(0.5*m.spend28.values+0.5*m.spend56.values))
print('blend .33/33/33:', mae((m.spend28+m.spend56+m.spend112)/3))
print('blend 28+ly:', mae(0.7*m.spend28.values+0.3*m.own_ly_spend4w.values))
print('blend 56+ly:', mae(0.7*m.spend56.values+0.3*m.own_ly_spend4w.values))
print('blend 112+ly:', mae(0.7*m.spend112.values+0.3*m.own_ly_spend4w.values))

# target stats by snapshot day (train)
g = tt.groupby('snapshot_day').future_spend_4w.agg(['mean','median',lambda s:(s==0).mean()])
print('\nby snapshot day:\n', g.round(2))

# zero/nonzero split: does spend28 predict zeros?
z = m.future_spend_4w==0
print('\nzero rows: spend28 mean', round(m.spend28[z].mean(),2), ' nonzero rows:', round(m.spend28[~z].mean(),2))
print('spend28==0 share among zeros:', round((m.spend28[z]==0).mean(),3))
print('P(spend28==0) -> target zero:', round(m[m.spend28==0].future_spend_4w.eq(0).mean(),3))

# quick numpy ridge on season table (numeric only) to gauge linear signal
num = [c for c in season.columns if c not in ('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(m[c])]
X = m[num].fillna(0).values.astype(float)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
mu, sd = X.mean(0), X.std(0)+1e-9
Xs = (X-mu)/sd
# add log1p versions of spend features
sp_cols = [i for i,c in enumerate(num) if c.startswith('spend') or c in ('lt_spend','own_ly_spend4w','own_ly2_spend4w')]
Xl = np.hstack([Xs, np.log1p(np.clip(X[:,sp_cols],0,None))])
def ridge(Xt, yt, lam=100.0):
    d = Xt.shape[1]
    A = Xt.T@Xt + lam*np.eye(d)
    return np.linalg.solve(A, Xt.T@yt)
w = ridge(Xl, y)
pred = Xl@w
print('\nridge(train-fit) MAE on train:', mae(pred))
w2 = ridge(Xl, np.log1p(y)); pred2 = np.expm1(Xl@w2)
print('ridge log-target MAE on train:', mae(pred2))
