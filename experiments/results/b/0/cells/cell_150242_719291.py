import warnings; warnings.filterwarnings('ignore')
t = load_saved('e007_lagseq.parquet')
tt = train_targets()
m = tt.merge(t, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values
b2 = (0.4*m.spend28.fillna(0)+0.3*m.spend112.fillna(0)/4+0.3*m.lag_mean_5_8.fillna(0)).values
res = y-b2
# which features predict the residual (positive residual = underprediction)
num = m.select_dtypes(include=[np.number])
num = num.loc[:, num.std()>0]
cc = num.corrwith(pd.Series(res)).drop('future_spend_4w', errors='ignore').sort_values()
print('corr with residual, top:'); print(cc.tail(15).round(3))
print('bottom:'); print(cc.head(6).round(3))
# active vs inactive households (spend28>0)
act = m.spend28.fillna(0)>0
print('n active', act.sum(), 'MAE active', np.abs(b2[act]-y[act]).mean().round(2))
print('MAE inactive', np.abs(b2[~act]-y[~act]).mean().round(2))
# among actives, corr of features with residual
cc2 = num[act].corrwith(pd.Series(res[act])).drop('future_spend_4w', errors='ignore').sort_values()
print('active resid corr top:'); print(cc2.tail(12).round(3))
print('active resid corr bottom:'); print(cc2.head(8).round(3))
# lag1 = last 4-week spend; how often is next 4w spend near lag1?
print('median |y - lag1|', np.abs(y - m.lag_spend_1.fillna(0)).mean().round(2))
# distribution of y conditional on zero28
print(m.groupby('zero28').future_spend_4w.describe().T.round(1))