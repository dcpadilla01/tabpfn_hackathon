df = load_saved('e010_decay.parquet')
tt = train_targets()
m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = m['future_spend_4w']

num = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ('snapshot_day',) and m[c].std()>0]
cor = m[num].corrwith(y).sort_values()
print('TOP positive:'); print(cor.tail(20).round(3))
print('\nTOP negative:'); print(cor.head(15).round(3))

# rank correlation (spearman) may reveal monotone tail predictors
sp = m[num].corrwith(y.rank(), method='spearman').sort_values()
print('\nSpearman top:'); print(sp.tail(15).round(3))
print('\nSpearman bottom:'); print(sp.head(10).round(3))
