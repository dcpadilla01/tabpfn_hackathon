import warnings; warnings.filterwarnings('ignore')
t = load_saved('e007_lagseq.parquet')
tt = train_targets()
m = tt.merge(t, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values
b2 = (0.4*m.spend28.fillna(0)+0.3*m.spend112.fillna(0)/4+0.3*m.lag_mean_5_8.fillna(0)).values
act = m.spend28.fillna(0)>0
# active-only: best linear combo of the big spend features (ridge, small grid)
feats = ['spend28','spend56','spend112','spend364','lag_mean_1_4','lag_mean_5_8','lt_spend','rwspend84','trips28','nprod28','iv_mean112','recency','tenure']
Xa = m.loc[act, feats].fillna(0).values
ya = y[act]
mu=Xa.mean(0); sd=Xa.std(0)+1e-9; Xs=(Xa-mu)/sd
best=None
for lam in [1,10,30,100,300]:
    A=Xs.T@Xs+lam*np.eye(len(feats)); w=np.linalg.solve(A, Xs.T@ya)
    p=Xs@w; mae=np.abs(p-ya).mean()
    print(lam, mae.round(2))
    if best is None or mae<best[1]: best=(lam,mae,w.copy())
w=best[2]
print('best lam', best[0], 'MAE', best[1].round(2))
print(dict(zip(feats,w.round(1))))
# add lag_ratio_1_13 and trend feats
feats2 = feats + ['lag_ratio_1_13','lag_trend_12_34','trend_112_364','spend_yoy28','weekend_share28','red_rate','tA_act','tB_act','tC_act','ndept28','toptrip_share28','demo_hhsize','demo_kids','has_demo']
Xa2 = m.loc[act, feats2].fillna(0).values
mu=Xa2.mean(0); sd=Xa2.std(0)+1e-9; Xs2=(Xa2-mu)/sd
for lam in [10,30,100]:
    A=Xs2.T@Xs2+lam*np.eye(len(feats2)); w2=np.linalg.solve(A, Xs2.T@ya)
    print(lam, np.abs(Xs2@w2-ya).mean().round(2))
# zero28 return prob: quick logistic-ish check via binned
zz = m[~act]
print('n zero28', len(zz))
# check corr of recency with P(y>0) among zero28 more finely + magnitude
zz2 = zz.copy(); zz2['ret'] = (zz2.future_spend_4w>0).astype(int)
print(zz2.groupby(pd.cut(zz2.recency, [0,7,14,21,28,56,100,200,400]), observed=True).ret.mean().round(2))