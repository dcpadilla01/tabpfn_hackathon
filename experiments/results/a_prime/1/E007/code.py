import numpy as np, pandas as pd, agent_api
t1 = agent_api.load_saved('e001_txhist.parquet'); t3 = agent_api.load_saved('e003_catmix.parquet')
print('e001', t1.shape, 'e003', t3.shape)
print('E001 cols:', list(t1.columns))
print('E003 extra:', [c for c in t3.columns if c not in t1.columns])
tt = agent_api.train_targets()
d = t3.merge(tt, on=['household_key','snapshot_day'])
y = d['future_spend_4w']
print('target:'); print(y.describe())
print('zero share:', round((y==0).mean(),3))
print('mean target by snapshot_day:'); print(d.groupby('snapshot_day')['future_spend_4w'].mean().round(1))
num = [c for c in d.columns if d[c].dtype.kind in 'ifbu' and c not in ('snapshot_day','future_spend_4w')]
cor = d[num].corrwith(y)
print('top |corr| with target:'); print(cor.reindex(cor.abs().sort_values(ascending=False).index).head(25).round(3))
spend28 = [c for c in t1.columns if 'spend' in c.lower() and '28' in c]
print('spend28-like cols:', spend28)
for c in spend28[:4]:
    print(c, 'proxy MAE:', round(np.abs(d[c]-y).mean(),2))
c0 = spend28[0] if spend28 else None
if c0:
    print('quantile profile of', c0); print(d.groupby(pd.qcut(d[c0],10,duplicates='drop'), observed=True)['future_spend_4w'].agg(['mean','count']))
    print('spend28==0 rows:', round((d[c0]==0).mean(),3)); print(d.loc[d[c0]==0,'future_spend_4w'].describe())


# ---- cell ----
import numpy as np, pandas as pd, agent_api
t1 = agent_api.load_saved('e001_txhist.parquet')
tr = agent_api.train_targets()
d = t1.merge(tr, on=['household_key','snapshot_day'])
y = d['future_spend_4w']
# spend_rate28 is spend in last 28d / 28 * 28 = spend_l1? check
print('corr spend_l1 vs spend_rate28:', d[['spend_l1','spend_rate28']].corr().iloc[0,1].round(4))
print('identical?', np.allclose(d['spend_l1'], d['spend_rate28']))
# distribution of spend_l1 vs y
for c in ['spend_l1','spend_l2','spend_l3','spend_l123_mean']:
    print(c, 'proxy MAE:', round(np.abs(d[c]-y).mean(),2))
# multiplicative structure: y/spend_l1 by spend_l1 quantile
q = pd.qcut(d['spend_l1'], 12, duplicates='drop')
prof = d.groupby(q, observed=True).apply(lambda g: pd.Series({
    'n': len(g), 'mean_y': g['future_spend_4w'].mean(), 'mean_l1': g['spend_l1'].mean(),
    'ratio': g['future_spend_4w'].mean()/max(g['spend_l1'].mean(),1e-9)}))
print(prof.round(3))
# zero-recent households: what do they buy next?
z = d[d['spend_l1']==0]
print('spend_l1==0:', len(z), 'mean y:', z['future_spend_4w'].mean().round(2), 'zero-y share:', (z['future_spend_4w']==0).mean().round(3))
print(z['future_spend_4w'].describe().round(2))
nz = d[d['spend_l1']>0]
print('spend_l1>0 mean y:', nz['future_spend_4w'].mean().round(2))
# does momentum help within spend_l1 bins?
d['bin'] = pd.qcut(d['spend_l1'], 10, duplicates='drop')
print(d.groupby('bin', observed=True)[['momentum','future_spend_4w']].mean().round(2))
# y vs spend_l1 log-log slope
m = (d['spend_l1']>0)&(y>0)
import numpy as np
sl = np.polyfit(np.log(d.loc[m,'spend_l1']), np.log(y[m]), 1)
print('log-log slope y~spend_l1:', sl.round(3))


# ---- cell ----
import numpy as np, pandas as pd, agent_api

def fn(view, snap):
    tx = view.transactions
    hh = pd.Index(view.households, name='household_key')
    tx = tx[tx.household_key.isin(hh)]
    def agg(lo, hi):  # window (snap-lo, snap-hi]
        m = tx[(tx.day > snap - lo) & (tx.day <= snap - hi)]
        g = m.groupby('household_key')
        return g.agg(sp=('sales_value','sum'), tr=('basket_id','nunique'),
                     pr=('product_id','nunique'))
    w1, w2, w3, w4 = agg(28,0), agg(56,28), agg(84,56), agg(112,84)
    w13 = agg(364,336)
    tot = tx.groupby('household_key').agg(sp=('sales_value','sum'), d0=('day','min'))
    idx = lambda df: df.reindex(hh).fillna({'sp':0.0,'tr':0,'pr':0})
    w1,w2,w3,w4,w13,tot = map(idx, (w1,w2,w3,w4,w13,tot))
    l1,l2,l3,l4,l13 = w1.sp,w2.sp,w3.sp,w4.sp,w13.sp
    l123 = (l1+l2+l3)/3.0
    l456 = (w4.sp + agg(140,112).sp.reindex(hh).fillna(0).sp + agg(168,140).sp.reindex(hh).fillna(0).sp)/3.0 if False else None
    # simpler: l456 from windows 4,5,6
    w5, w6 = agg(140,112), agg(168,140)
    w5, w6 = idx(w5), idx(w6)
    l456 = (l4 + w5.sp + w6.sp)/3.0
    tenure = (snap - tot.d0 + 1).clip(lower=1)
    longrun28 = tot.sp/tenure*28.0
    days_last = snap - tx.groupby('household_key').day.max()
    days_last = days_last.reindex(hh).fillna(9999)
    f = pd.DataFrame(index=hh)
    lg = lambda s: np.log1p(s.clip(lower=0))
    f['log_spend_l1'] = lg(l1); f['log_spend_l2'] = lg(l2); f['log_spend_l3'] = lg(l3)
    f['log_spend_l123'] = lg(l123); f['log_spend_l456'] = lg(l456)
    f['log_spend_l13'] = lg(l13); f['log_spend_total'] = lg(tot.sp)
    f['log_longrun28'] = lg(longrun28)
    f['log_trips_l1'] = lg(w1.tr.astype(float)); f['log_prods_l1'] = lg(w1.pr.astype(float))
    f['log_avg_basket_l1'] = lg(w1.sp/w1.tr.clip(lower=1))
    f['r_l1_l2'] = l1/(l2+1.0); f['r_l1_l123'] = l1/(l123+1.0)
    f['r_l1_longrun'] = l1/(longrun28+1.0); f['r_l123_longrun'] = l123/(longrun28+1.0)
    f['ewma_spend'] = (8*l1+4*l2+2*l3+l4)/15.0
    f['log_ewma_spend'] = lg(f['ewma_spend'])
    f['share_w1_of_123'] = l1/(l1+l2+l3+1.0)
    f['days_since_last_c'] = days_last.clip(upper=180).astype(float)
    return f

out = agent_api.build_features(fn)
print(out.shape); print(out.head(3).T)
base = agent_api.load_saved('e003_catmix.parquet')
m = base.merge(out.reset_index(), on=['household_key','snapshot_day'], how='inner')
print('merged', m.shape, 'nulls:', int(m.isna().sum().sum()))
p = agent_api.save_table(m, 'e007_logratio.parquet')
print(p)
