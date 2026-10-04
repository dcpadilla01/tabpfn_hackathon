
import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e011_rank.parquet')
print('e011 shape', t.shape)
print('e011 cols:', list(t.columns))

tt = agent_api.train_targets()
y = tt.future_spend_4w
print('targets n=%d zero=%.3f mean=%.1f med=%.1f p90=%.1f' % (len(tt), (y==0).mean(), y.mean(), y.median(), y.quantile(.9)))
print(tt.groupby('snapshot_day').future_spend_4w.agg(['count','mean','median']))

view = agent_api.snapshot()
tx = view.transactions
print('tx rows', len(tx))

TRAIN = [95,123,151,179,207,235,263,291,319,347,375,403,431]
feats = []
for sday in TRAIN:
    w1 = tx[(tx.day > sday-28) & (tx.day <= sday)]
    w2 = tx[(tx.day > sday-56) & (tx.day <= sday-28)]
    w3 = tx[(tx.day > sday-84) & (tx.day <= sday-56)]
    tot = w1.groupby('household_key').sales_value.sum()
    f = pd.DataFrame({'s28': tot})
    if len(w2):
        p2 = w2[['household_key','product_id']].drop_duplicates().assign(_in=1)
        m1 = w1.merge(p2, on=['household_key','product_id'], how='left')
        rep = m1[m1._in==1].groupby('household_key').sales_value.sum()
        f['rep12'] = (rep/tot).where(tot>0)
        p1s = w1[['household_key','product_id']].drop_duplicates().assign(_a=1)
        u = p1s.merge(p2, on=['household_key','product_id'], how='outer', indicator=True)
        f['jacc12'] = u[u._merge=='both'].groupby('household_key').size() / u.groupby('household_key').size()
    if len(w3):
        p3 = w3[['household_key','product_id']].drop_duplicates().assign(_in=1)
        m1 = w1.merge(p3, on=['household_key','product_id'], how='left')
        rep = m1[m1._in==1].groupby('household_key').sales_value.sum()
        f['rep13'] = (rep/tot).where(tot>0)
        if len(w2):
            p23 = w2.merge(p3, on=['household_key','product_id'])[['household_key','product_id']].drop_duplicates().assign(_in=1)
            m1 = w1.merge(p23, on=['household_key','product_id'], how='left')
            rep = m1[m1._in==1].groupby('household_key').sales_value.sum()
            f['rep123'] = (rep/tot).where(tot>0)
    # 56d repeat share: spend in (s-56,s] on products in (s-112,s-56]
    wa = tx[(tx.day > sday-56) & (tx.day <= sday)]
    wb = tx[(tx.day > sday-112) & (tx.day <= sday-56)]
    ta = wa.groupby('household_key').sales_value.sum()
    if len(wb):
        pb = wb[['household_key','product_id']].drop_duplicates().assign(_in=1)
        ma = wa.merge(pb, on=['household_key','product_id'], how='left')
        rep = ma[ma._in==1].groupby('household_key').sales_value.sum()
        f['rep56'] = (rep/ta).where(ta>0)
    gp = w1.groupby(['household_key','product_id']).sales_value.sum().reset_index()
    gp['rk'] = gp.groupby('household_key').sales_value.rank(ascending=False, method='first')
    f['conc5'] = gp[gp.rk<=5].groupby('household_key').sales_value.sum()/tot
    f['nprod1'] = gp.groupby('household_key').size()
    if len(w2):
        f['nprod2'] = w2[['household_key','product_id']].drop_duplicates().groupby('household_key').size()
    f['snapshot_day'] = sday
    feats.append(f.reset_index())

F = pd.concat(feats, ignore_index=True)
d = tt.merge(F, on=['household_key','snapshot_day'], how='left')
y = d.future_spend_4w
pred = d.s28.fillna(0)
print('MAE s28 alone: %.2f | s28*0.9: %.2f | s28*1.1: %.2f' % ((pred-y).abs().mean(), (pred*0.9-y).abs().mean(), (pred*1.1-y).abs().mean()))
z = y==0
print('y==0: share %.3f mean_s28 %.1f MAE %.2f || y>0: MAE %.2f' % (z.mean(), pred[z].mean(), (pred[z]-y[z]).abs().mean(), (pred[~z]-y[~z]).abs().mean()))
for c in ['rep12','rep13','rep123','jacc12','rep56','conc5','nprod1','nprod2']:
    if c in d: print(c, 'corr(y) %.3f  corr|y>0| %.3f' % (d[c].corr(y), d[c][~z].corr(y[~z])))
d['bin'] = pd.cut(d.s28.fillna(-1), [-1.1,-0.5,10,25,50,100,200,1e9])
for c in ['rep12','jacc12','rep56']:
    cs = d.groupby('bin', observed=True).apply(lambda g: g[c].corr(g.future_spend_4w) if g[c].notna().sum()>30 else np.nan)
    print('within-s28-bin corr', c, cs.round(3).to_dict())
d['rq'] = pd.qcut(d.rep12, 4, duplicates='drop')
print(d.groupby('rq', observed=True).agg(my=('future_spend_4w','mean'), n=('future_spend_4w','size'), ms28=('s28','mean')))
