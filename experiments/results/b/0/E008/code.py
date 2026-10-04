t = load_saved('e007_lagseq.parquet')
print('shape', t.shape)
cols = list(t.columns)
print('n cols', len(cols))
print(cols)
tt = train_targets()
print(tt.shape)
print(tt.future_spend_4w.describe())
print('zero share', (tt.future_spend_4w==0).mean())
m = tt.merge(t, on=['household_key','snapshot_day'], how='left')
num = m.select_dtypes(include=[np.number])
cor = num.corrwith(m.future_spend_4w)
cor = cor.drop('future_spend_4w', errors='ignore').sort_values()
print('TOP positive:'); print(cor.tail(20))
print('TOP negative:'); print(cor.head(8))
v = snapshot()
print('households type', type(v.households), len(v.households))
print('txn shape', v.transactions.shape, 'max day', v.transactions.day.max())
print('has products', hasattr(v,'products'), 'has demographics', hasattr(v,'demographics'), 'has campaigns', hasattr(v,'campaigns'), 'has dm', hasattr(v,'display_mailer'), 'has cr', hasattr(v,'coupon_redemptions'))


# ---- cell ----
t = load_saved('e007_lagseq.parquet')
d = [c for c in t.columns if c.startswith('dsp_')]
print('dsp cols', len(d))
print(t[d].notna().sum().sum(), 'non-nan total')
print(t[d].describe().T.head(5))
m = train_targets().merge(t, on=['household_key','snapshot_day'], how='left')
num = m.select_dtypes(include=[np.number]).drop(columns=['dsp_ '], errors='ignore')
cor = num.corrwith(m.future_spend_4w).drop('future_spend_4w', errors='ignore').sort_values()
print('TOP positive:'); print(cor.tail(25).round(3))
print('neg:'); print(cor.head(6).round(3))
v = snapshot()
print([a for a in dir(v) if not a.startswith('_')])


# ---- cell ----
import warnings; warnings.filterwarnings('ignore')
t = load_saved('e007_lagseq.parquet')
m = train_targets().merge(t, on=['household_key','snapshot_day'], how='left')
num = m.select_dtypes(include=[np.number])
num = num.loc[:, num.std() > 0]  # drop zero-variance
cor = num.corrwith(m.future_spend_4w).drop('future_spend_4w', errors='ignore').sort_values()
print('TOP+ :'); print(cor.tail(18).round(3))
print('TOP- :'); print(cor.head(10).round(3))
# check lag_spend_13 vs own_ly_spend4w
print('corr lag13 vs own_ly:', m[['lag_spend_13','own_ly_spend4w']].corr().iloc[0,1].round(3))
print('corr lag1 vs spend28:', m[['lag_spend_1','spend28']].corr().iloc[0,1].round(3))
# tenure dist by snapshot
print(m.groupby('snapshot_day').tenure.mean().round(0))
print(m.groupby('snapshot_day').future_spend_4w.mean().round(1))
p = snapshot().products
print('products', p.shape, p.columns.tolist())
print('n commodities', p.commodity_desc.nunique(), 'n depts', p.department.nunique())
print(p.brand.value_counts().head())


# ---- cell ----
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

# ---- cell ----
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

# ---- cell ----
import warnings; warnings.filterwarnings('ignore')
t = load_saved('e007_lagseq.parquet')
tt = train_targets()
m = tt.merge(t, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values
b2 = (0.4*m.spend28.fillna(0)+0.3*m.spend112.fillna(0)/4+0.3*m.lag_mean_5_8.fillna(0)).values
act = m.spend28.fillna(0)>0
# For actives: how much better can a 2-piece linear model do? quick ridge on a few feats
feats = ['spend28','spend56','spend112','spend364','lag_mean_1_4','lag_mean_5_8','trips28','nprod28','recency','iv_mean112','trend_112_364','lt_spend','tenure']
X = m[feats].fillna(0).values
Xa = X[act]; ya = y[act]; ba = b2[act]
# standardize
mu = Xa.mean(0); sd = Xa.std(0)+1e-9
Xs = (Xa-mu)/sd
from numpy.linalg import lstsq
# ridge via normal equations
lam=1.0
A = Xs.T@Xs + lam*np.eye(len(feats)); bvec = Xs.T@(ya-ba)
w = np.linalg.solve(A,bvec)
pred = ba + Xs@w
print('active MAE blend3', np.abs(ba-ya).mean().round(2), '-> ridge resid corr feats', np.abs(pred-ya).mean().round(2))
print(dict(zip(feats,w.round(1))))
# also check zero-inflation: P(y>0) model — proxy by spend28>0 already; among zero28, P(y>0)?
z = ~act
print('P(y>0 | zero28)', (y[z]>0).mean().round(3))
# among zero28 with y>0, magnitude
print('y | zero28 & y>0: median', np.median(y[z&(y>0)]).round(1), 'mean', y[z&(y>0)].mean().round(1))
# does recency predict return among zero28?
zz = m[z]
for c in ['recency','lag_zero_streak','iv_mean112','lt_spend','spend112']:
    a = zz.groupby(pd.qcut(zz[c], 5, duplicates='drop'), observed=True).future_spend_4w.apply(lambda s:(s>0).mean())
    print(c, a.round(2).values)

# ---- cell ----
import warnings; warnings.filterwarnings('ignore')
v = snapshot()
dm = v.display_mailer
print(dm.shape, dm.columns.tolist())
print(dm.head())
print(dm.display.value_counts())
print(dm.mailer.value_counts())
print('weeks', dm.week_no.min(), dm.week_no.max())
print('n products', dm.product_id.nunique(), 'n stores', dm.store_id.nunique())
# how many households' recent purchases have display/mailer info?
tx = v.transactions
print('tx max day', tx.day.max(), 'shape', tx.shape)
print(tx[['quantity','sales_value','coupon_disc','coupon_match_disc','retail_disc','trans_time']].describe().T.round(2))
# check zero-variance dsp cols in full table
t = load_saved('e007_lagseq.parquet')
d = [c for c in t.columns if c.startswith('dsp_')]
vv = t[d].var()
print('zero-var dsp cols:', [c for c in d if vv[c]==0])

# ---- cell ----
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

# ---- cell ----
import warnings; warnings.filterwarnings('ignore')
v = snapshot()
dm = v.display_mailer
tx = v.transactions
# join recent tx (last 28d) with display/mailer for that product/store/week
last28 = tx[tx.day >= 459-27]
print('last28 rows', len(last28))
dm_small = dm[['product_id','store_id','week_no','display','mailer']]
j = last28.merge(dm_small, on=['product_id','store_id','week_no'], how='left')
print('match rate', j.display.notna().mean().round(3))
print('display dist among matched', j.display.value_counts(dropna=False).head(8))
print('mailer dist among matched', j.mailer.value_counts(dropna=False).head(8))
# per-household exposure
j['has_disp'] = (j.display.notna() & (j.display!=0)).astype(int)
j['has_mail'] = (j.mailer.notna() & (j.mailer!='0')).astype(int)
g = j.groupby('household_key').agg(disp_rows=('has_disp','sum'), mail_rows=('has_mail','sum'), rows=('has_disp','size'))
print(g.describe().T.round(2))
# spend on displayed products vs not, last 28d
print('spend share on displayed rows', (j.loc[j.has_disp==1,'sales_value'].sum()/j.sales_value.sum()).round(3))
print('spend share on mailer rows', (j.loc[j.has_mail==1,'sales_value'].sum()/j.sales_value.sum()).round(3))

# ---- cell ----
import warnings; warnings.filterwarnings('ignore')
v = snapshot()
tx = v.transactions
hh = list(v.households)
print('n hh', len(hh))
g = tx.groupby('household_key').sales_value.sum()
print('groupby ok', len(g))
dm = v.display_mailer
dm_small = dm[['product_id','store_id','week_no','display','mailer']]
last28 = tx[tx.day >= 459-27]
j = last28.merge(dm_small, on=['product_id','store_id','week_no'], how='left')
print('merge ok', j.shape, 'match rate', j.display.notna().mean().round(3))
print('ct', v.campaign_targets.shape, 'cr', v.coupon_redemptions.shape, 'camps', v.campaigns.shape)
print(v.campaign_targets.description.value_counts())
print(v.demographics.shape)

# ---- cell ----
import warnings; warnings.filterwarnings('ignore')
v = snapshot()
tx = v.transactions
hh = v.households
print('n hh', type(hh), hh is None)
print('len hh', len(hh) if hh is not None else None)

# ---- cell ----
import warnings; warnings.filterwarnings('ignore')
v = snapshot()
tx = v.transactions
dm = v.display_mailer
dm_small = dm[['product_id','store_id','week_no','display','mailer']]
last28 = tx[tx.day >= 459-27]
j = last28.merge(dm_small, on=['product_id','store_id','week_no'], how='left')
print('merge ok', j.shape, 'match rate', j.display.notna().mean().round(3))
print('n hh in last28', j.household_key.nunique())
print('ct', v.campaign_targets.shape, 'cr', v.coupon_redemptions.shape, 'camps', v.campaigns.shape)
print(v.campaign_targets.description.value_counts())
print('demo', v.demographics.shape)
print('hh via tx', tx.household_key.nunique())

# ---- cell ----
import warnings; warnings.filterwarnings('ignore')

def fn(view, snapshot_day):
    e7 = load_saved('e007_lagseq.parquet')
    sub = e7.loc[e7.snapshot_day == snapshot_day].set_index('household_key').drop(columns=['snapshot_day'])
    tx = view.transactions
    w0 = snapshot_day - 27; w112 = snapshot_day - 111
    last28 = tx[tx.day >= w0]
    last112 = tx[tx.day >= w112]
    spend28_hh = last28.groupby('household_key').sales_value.sum()
    spend112_hh = last112.groupby('household_key').sales_value.sum()
    out = pd.DataFrame(index=sub.index)
    # display / mailer exposure on recently purchased items
    wk0 = (w0 + 8) // 7; wk1 = (snapshot_day + 8) // 7
    dm = view.display_mailer
    dms = dm.loc[(dm.week_no >= wk0) & (dm.week_no <= wk1), ['product_id','store_id','week_no','display','mailer']]
    j = last28.merge(dms, on=['product_id','store_id','week_no'], how='left')
    disp = j.display.notna() & (j.display != 0)
    mail = j.mailer.notna() & (j.mailer != '0')
    out['disp_spend28'] = j.loc[disp].groupby('household_key').sales_value.sum()
    out['mail_spend28'] = j.loc[mail].groupby('household_key').sales_value.sum()
    out['disp_share28'] = out['disp_spend28'] / spend28_hh
    out['mail_share28'] = out['mail_spend28'] / spend28_hh
    out['disp_rows28'] = j.loc[disp].groupby('household_key').size()
    out['mail_rows28'] = j.loc[mail].groupby('household_key').size()
    out['nlines28'] = j.groupby('household_key').size()
    # private-brand share of last-28d spend
    br = view.products[['product_id','brand']]
    jb = last28.merge(br, on='product_id', how='left')
    priv = jb.loc[jb.brand == 'Private'].groupby('household_key').sales_value.sum()
    out['brand_priv_spend28'] = priv
    out['brand_priv_share28'] = priv / spend28_hh
    # repeat-commodity spend (commodity bought before within 112d window)
    cm = view.products[['product_id','commodity_desc']]
    j3 = last112.merge(cm, on='product_id', how='left')
    first_d = j3.groupby(['household_key','commodity_desc']).day.transform('min')
    ret = j3[j3.day > first_d]
    out['dsp_rets112'] = ret.groupby('household_key').sales_value.sum()
    out['dsp_rets28'] = ret.loc[ret.day >= w0].groupby('household_key').sales_value.sum()
    out['dsp_rets_share112'] = out['dsp_rets112'] / spend112_hh
    return out

fe = build_features(fn)
print(fe.shape)
print(fe.head(3).T.head(30))
print(fe.groupby('snapshot_day').size())
p = save_table(fe, 'e008_exposure')
print(p)