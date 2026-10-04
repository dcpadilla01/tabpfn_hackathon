df = load_saved('e010_decay.parquet')
tt = train_targets()
m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('train rows:', len(m), 'val rows:', len(df)-len(m))
y = m['future_spend_4w']
print('target: mean %.1f median %.1f p90 %.1f p99 %.1f max %.1f, zero share %.3f' % (y.mean(), y.median(), y.quantile(.9), y.quantile(.99), y.max(), (y==0).mean()))

# simple predictor baselines on train rows
for c in ['lag_spend_1','spend28','lag_mean_1_4','dec_spend14','rwspend84']:
    pred = m[c].fillna(0).clip(lower=0)
    print('MAE %s: %.2f' % (c, (pred-y).abs().mean()))
print('MAE median: %.2f' % (y.median()-y).abs().mean())
print('MAE blend lag1*0.6+lag2*0.25+lag3*0.15: %.2f' % ((0.6*m.lag_spend_1+0.25*m.lag_spend_2+0.15*m.lag_spend_3).fillna(0).sub(y).abs().mean()))

# correlation of features with target
num = df.select_dtypes(include=[np.number]).columns.tolist()
num = [c for c in num if c not in ('snapshot_day',)]
cor = m[num].corrwith(y).sort_values()
print('\nmost negative corr:'); print(cor.head(8).round(3))
print('\nmost positive corr:'); print(cor.tail(15).round(3))

# MAE by target bucket for the best single predictor
pred = (0.6*m.lag_spend_1+0.25*m.lag_spend_2+0.15*m.lag_spend_3).fillna(0)
b = pd.qcut(y, 5, duplicates='drop')
print('\nMAE by target quintile (blend pred):')
print(m.groupby(b, observed=True).apply(lambda g: pd.Series({'n':len(g),'y_med':g.future_spend_4w.median(),'mae':(g.future_spend_4w-pred[g.index]).abs().mean()})))
