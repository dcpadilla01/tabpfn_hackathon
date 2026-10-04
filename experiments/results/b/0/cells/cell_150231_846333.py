import warnings; warnings.filterwarnings('ignore')
t = load_saved('e007_lagseq.parquet')
tt = train_targets()
m = tt.merge(t, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values
def mae(p): return np.abs(p-y).mean()
print('mean pred', mae(np.full(len(y), y.mean())).round(2))
for c in ['spend28','spend56','spend112','lag_mean_1_4','lag_spend_1','rwspend84','spend_per_day28','lt_spend']:
    print(c, mae(m[c].fillna(0).values).round(2))
# blends
b = 0.5*m.spend28.fillna(0)+0.5*m.spend112.fillna(0)/4
print('blend28/112', mae(b.values).round(2))
b2 = 0.4*m.spend28.fillna(0)+0.3*m.spend112.fillna(0)/4+0.3*m.lag_mean_5_8.fillna(0)
print('blend3', mae(b2.values).round(2))
# residual analysis of blend3
res = y - b2.values
print('resid mean', res.mean().round(2), 'MAE', np.abs(res).mean().round(2))
q = pd.qcut(b2.values, 10, duplicates='drop')
g = pd.DataFrame({'q':q,'y':y,'p':b2.values}).groupby('q', observed=True).agg(p_mean=('p','mean'), y_mean=('y','mean'), y_med=('y','median'), n=('y','size'), mae=('y', lambda s: None))
g['bias'] = g.y_mean-g.p_mean
print(g.round(1))
# MAE by tenure
m['ten_b'] = pd.cut(m.tenure, [84,120,200,300,500])
print(m.groupby('ten_b', observed=True).apply(lambda d: pd.Series({'n':len(d), 'mae_blend3': np.abs(d.future_spend_4w-b2[d.index]).mean()}), include_groups=False).round(1))
# zero rows: what pred is best for them
z = m.future_spend_4w==0
print('zero rows MAE blend3', np.abs(b2.values[z]-y[z]).mean().round(2), 'n', z.sum())