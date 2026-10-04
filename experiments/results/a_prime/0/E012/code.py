
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


# ---- cell ----

import agent_api, pandas as pd, numpy as np

tx_all = agent_api.snapshot().transactions
TRAIN = [95,123,151,179,207,235,263,291,319,347,375,403,431]

def habit_feats(tx, sday):
    w1 = tx[(tx.day > sday-28) & (tx.day <= sday)]
    w2 = tx[(tx.day > sday-56) & (tx.day <= sday-28)]
    w3 = tx[(tx.day > sday-84) & (tx.day <= sday-56)]
    w4 = tx[(tx.day > sday-112) & (tx.day <= sday-84)]
    tot = w1.groupby('household_key').sales_value.sum()
    f = pd.DataFrame(index=tot.index)
    def repshare(win, proddf, name):
        if len(win)==0: return
        p = proddf[['household_key','product_id']].drop_duplicates().assign(_in=1)
        m = win.merge(p, on=['household_key','product_id'], how='left')
        r = m[m._in==1].groupby('household_key').sales_value.sum()
        f[name] = (r/tot).where(tot>0)
    p1 = w1[['household_key','product_id']].drop_duplicates().assign(_a=1)
    if len(w2):
        p2 = w2[['household_key','product_id']].drop_duplicates().assign(_in=1)
        repshare(w1, p2, 'rep12')
        u = p1.merge(p2, on=['household_key','product_id'], how='outer', indicator=True)
        f['jacc12'] = u[u._merge=='both'].groupby('household_key').size()/u.groupby('household_key').size()
        if len(w3):
            p3 = w3[['household_key','product_id']].drop_duplicates().assign(_in=1)
            repshare(w1, p3, 'rep13')
            p23 = w2.merge(p3, on=['household_key','product_id'])[['household_key','product_id']].drop_duplicates().assign(_in=1)
            repshare(w1, p23, 'rep123')
    if len(w4):
        p4 = w4[['household_key','product_id']].drop_duplicates().assign(_in=1)
        wa = tx[(tx.day > sday-56) & (tx.day <= sday)]
        repshare(wa, p4, 'rep56')
    gp = w1.groupby(['household_key','product_id']).sales_value.sum().reset_index()
    gp['rk'] = gp.groupby('household_key').sales_value.rank(ascending=False, method='first')
    f['conc5'] = gp[gp.rk<=5].groupby('household_key').sales_value.sum()/tot
    f['nprod1'] = gp.groupby('household_key').size()
    if len(w2): f['nprod2'] = w2[['household_key','product_id']].drop_duplicates().groupby('household_key').size()
    if len(w3): f['nprod3'] = w3[['household_key','product_id']].drop_duplicates().groupby('household_key').size()
    past = tx[tx.day <= sday-28]
    gaps = past.sort_values('day').groupby(['household_key','product_id']).day.apply(lambda s: s.diff().median())
    due = gaps[gaps < 70].reset_index().assign(_in=1)
    m = w1.merge(due, on=['household_key','product_id'], how='left')
    r = m[m._in==1].groupby('household_key').sales_value.sum()
    f['due70'] = (r/tot).where(tot>0)
    if 'rep12' in f: f['rep12_amt'] = f['rep12']*tot
    return f.reset_index()

rows=[]
for sday in TRAIN:
    f = habit_feats(tx_all, sday); f['snapshot_day']=sday; rows.append(f)
H = pd.concat(rows, ignore_index=True)

base = agent_api.load_saved('e011_rank.parquet')
tt = agent_api.train_targets()
d = tt.merge(base.drop(columns=['household_key','snapshot_day']), left_index=True, right_index=True) \
      .merge(H, on=['household_key','snapshot_day'], how='left')
print('merged', d.shape)

y = d.future_spend_4w.values
tr = (d.snapshot_day <= 403).values; va = (d.snapshot_day == 431).values
num = d.select_dtypes(include=[np.number]).drop(columns=['snapshot_day','future_spend_4w'])

def fit_eval(cols, tag, alpha=10.0):
    X = d[cols].astype(float)
    med = X.median()
    X = X.fillna(med).values
    mu = X[tr].mean(0); sd = X[tr].std(0)+1e-9
    Xz = (X-mu)/sd
    A = np.hstack([np.ones((len(Xz),1)), Xz])
    lam = alpha/len(A[tr])
    I = np.eye(A.shape[1]); I[0,0]=0
    w = np.linalg.solve(A[tr].T@A[tr] + lam*np.sum(np.abs(A[tr])**2,0)[:,None]*I, A[tr].T@y[tr])
    p = A[va]@w
    mae = np.abs(p-y[va]).mean()
    print(tag, 'inner MAE %.2f (nfeat %d)' % (mae, len(cols)))
    return mae

bcols = [c for c in num.columns if not c.startswith(('rep','jac','conc','nprod','due'))]
hcols = [c for c in num.columns if c.startswith(('rep','jac','conc','nprod','due'))]
fit_eval(bcols, 'base only')
fit_eval(bcols+hcols, 'base+habit')
fit_eval(hcols, 'habit only')
d['s28_x_rep12'] = d.spend_28*d.rep12.fillna(0)
d['s28_x_due'] = d.spend_28*d.due70.fillna(0)
num2 = d.select_dtypes(include=[np.number]).drop(columns=['snapshot_day','future_spend_4w'])
b2 = [c for c in num2.columns if not c.startswith(('rep','jac','conc','nprod','due'))]
fit_eval(b2, 'base + s28 interactions')
fit_eval(b2+hcols, 'base + interactions + habit')
print('habit cols:', hcols)


# ---- cell ----

import agent_api, pandas as pd, numpy as np

tx_all = agent_api.snapshot().transactions
TRAIN = [95,123,151,179,207,235,263,291,319,347,375,403,431]

# Precompute product-level median gap ONCE (household x product), using full tx up to 459.
txs = tx_all.sort_values(['household_key','product_id','day'])
g = txs.groupby(['household_key','product_id'], sort=False).day
dif = g.diff()
med_gap = dif.groupby([txs.household_key, txs.product_id]).median()
print('med_gap computed', len(med_gap))

def habit_feats(tx, sday, med_gap):
    w1 = tx[(tx.day > sday-28) & (tx.day <= sday)]
    w2 = tx[(tx.day > sday-56) & (tx.day <= sday-28)]
    w3 = tx[(tx.day > sday-84) & (tx.day <= sday-56)]
    w4 = tx[(tx.day > sday-112) & (tx.day <= sday-84)]
    tot = w1.groupby('household_key').sales_value.sum()
    f = pd.DataFrame(index=tot.index)
    def repshare(win, proddf, name):
        if len(win)==0: return
        p = proddf[['household_key','product_id']].drop_duplicates().assign(_in=1)
        m = win.merge(p, on=['household_key','product_id'], how='left')
        r = m[m._in==1].groupby('household_key').sales_value.sum()
        f[name] = (r/tot).where(tot>0)
    p1 = w1[['household_key','product_id']].drop_duplicates().assign(_a=1)
    if len(w2):
        p2 = w2[['household_key','product_id']].drop_duplicates().assign(_in=1)
        repshare(w1, p2, 'rep12')
        u = p1.merge(p2, on=['household_key','product_id'], how='outer', indicator=True)
        f['jacc12'] = u[u._merge=='both'].groupby('household_key').size()/u.groupby('household_key').size()
        if len(w3):
            p3 = w3[['household_key','product_id']].drop_duplicates().assign(_in=1)
            repshare(w1, p3, 'rep13')
            p23 = w2.merge(p3, on=['household_key','product_id'])[['household_key','product_id']].drop_duplicates().assign(_in=1)
            repshare(w1, p23, 'rep123')
    if len(w4):
        p4 = w4[['household_key','product_id']].drop_duplicates().assign(_in=1)
        wa = tx[(tx.day > sday-56) & (tx.day <= sday)]
        repshare(wa, p4, 'rep56')
    gp = w1.groupby(['household_key','product_id']).sales_value.sum().reset_index()
    gp['rk'] = gp.groupby('household_key').sales_value.rank(ascending=False, method='first')
    f['conc5'] = gp[gp.rk<=5].groupby('household_key').sales_value.sum()/tot
    f['nprod1'] = gp.groupby('household_key').size()
    if len(w2): f['nprod2'] = w2[['household_key','product_id']].drop_duplicates().groupby('household_key').size()
    if len(w3): f['nprod3'] = w3[['household_key','product_id']].drop_duplicates().groupby('household_key').size()
    # due-repurchase using precomputed gaps (gap computed on data <= sday-28 to avoid leakage)
    past = tx[tx.day <= sday-28]
    mg = med_gap.reindex(pd.MultiIndex.from_frame(past[['household_key','product_id']].drop_duplicates()))
    due = (mg < 70).dropna()
    due = due[due].reset_index()[['household_key','product_id']].assign(_in=1)
    m = w1.merge(due, on=['household_key','product_id'], how='left')
    r = m[m._in==1].groupby('household_key').sales_value.sum()
    f['due70'] = (r/tot).where(tot>0)
    if 'rep12' in f: f['rep12_amt'] = f['rep12']*tot
    return f.reset_index()

rows=[]
for sday in TRAIN:
    f = habit_feats(tx_all, sday, med_gap); f['snapshot_day']=sday; rows.append(f)
H = pd.concat(rows, ignore_index=True)
print('H', H.shape, list(H.columns))

base = agent_api.load_saved('e011_rank.parquet')
tt = agent_api.train_targets()
d = tt.merge(base.drop(columns=['household_key','snapshot_day']), left_index=True, right_index=True) \
      .merge(H, on=['household_key','snapshot_day'], how='left')
print('merged', d.shape)

y = d.future_spend_4w.values
tr = (d.snapshot_day <= 403).values; va = (d.snapshot_day == 431).values
num = d.select_dtypes(include=[np.number]).drop(columns=['snapshot_day','future_spend_4w'])

def fit_eval(cols, tag, alpha=10.0):
    X = d[cols].astype(float)
    X = X.fillna(X.median()).values
    mu = X[tr].mean(0); sd = X[tr].std(0)+1e-9
    Xz = (X-mu)/sd
    A = np.hstack([np.ones((len(Xz),1)), Xz])
    lam = alpha/len(A[tr])
    I = np.eye(A.shape[1]); I[0,0]=0
    w = np.linalg.solve(A[tr].T@A[tr] + lam*np.sum(np.abs(A[tr])**2,0)[:,None]*I, A[tr].T@y[tr])
    p = A[va]@w
    mae = np.abs(p-y[va]).mean()
    print(tag, 'inner MAE %.2f (nfeat %d)' % (mae, len(cols)))
    return mae

bcols = [c for c in num.columns if not c.startswith(('rep','jac','conc','nprod','due'))]
hcols = [c for c in num.columns if c.startswith(('rep','jac','conc','nprod','due'))]
fit_eval(bcols, 'base only')
fit_eval(bcols+hcols, 'base+habit')
fit_eval(hcols, 'habit only')
d['s28_x_rep12'] = d.spend_28*d.rep12.fillna(0)
d['s28_x_due'] = d.spend_28*d.due70.fillna(0)
num2 = d.select_dtypes(include=[np.number]).drop(columns=['snapshot_day','future_spend_4w'])
b2 = [c for c in num2.columns if not c.startswith(('rep','jac','conc','nprod','due'))]
fit_eval(b2, 'base + s28 interactions')
fit_eval(b2+hcols, 'base + interactions + habit')
print('habit cols:', hcols)


# ---- cell ----

import agent_api, pandas as pd, numpy as np

tx_all = agent_api.snapshot().transactions
TRAIN = [95,123,151,179,207,235,263,291,319,347,375,403,431]
txs = tx_all.sort_values(['household_key','product_id','day'])
dif = txs.groupby(['household_key','product_id'], sort=False).day.diff()
med_gap = dif.groupby([txs.household_key, txs.product_id]).median()

def habit_feats(tx, sday, med_gap):
    w1 = tx[(tx.day > sday-28) & (tx.day <= sday)]
    w2 = tx[(tx.day > sday-56) & (tx.day <= sday-28)]
    w3 = tx[(tx.day > sday-84) & (tx.day <= sday-56)]
    w4 = tx[(tx.day > sday-112) & (tx.day <= sday-84)]
    tot = w1.groupby('household_key').sales_value.sum()
    f = pd.DataFrame(index=tot.index)
    def repshare(win, proddf, name):
        if len(win)==0: return
        p = proddf[['household_key','product_id']].drop_duplicates().assign(_in=1)
        m = win.merge(p, on=['household_key','product_id'], how='left')
        r = m[m._in==1].groupby('household_key').sales_value.sum()
        f[name] = (r/tot).where(tot>0)
    p1 = w1[['household_key','product_id']].drop_duplicates().assign(_a=1)
    if len(w2):
        p2 = w2[['household_key','product_id']].drop_duplicates().assign(_in=1)
        repshare(w1, p2, 'rep12')
        u = p1.merge(p2, on=['household_key','product_id'], how='outer', indicator=True)
        f['jacc12'] = u[u._merge=='both'].groupby('household_key').size()/u.groupby('household_key').size()
        if len(w3):
            p3 = w3[['household_key','product_id']].drop_duplicates().assign(_in=1)
            repshare(w1, p3, 'rep13')
            p23 = w2.merge(p3, on=['household_key','product_id'])[['household_key','product_id']].drop_duplicates().assign(_in=1)
            repshare(w1, p23, 'rep123')
    if len(w4):
        p4 = w4[['household_key','product_id']].drop_duplicates().assign(_in=1)
        wa = tx[(tx.day > sday-56) & (tx.day <= sday)]
        repshare(wa, p4, 'rep56')
    gp = w1.groupby(['household_key','product_id']).sales_value.sum().reset_index()
    gp['rk'] = gp.groupby('household_key').sales_value.rank(ascending=False, method='first')
    f['conc5'] = gp[gp.rk<=5].groupby('household_key').sales_value.sum()/tot
    f['nprod1'] = gp.groupby('household_key').size()
    if len(w2): f['nprod2'] = w2[['household_key','product_id']].drop_duplicates().groupby('household_key').size()
    if len(w3): f['nprod3'] = w3[['household_key','product_id']].drop_duplicates().groupby('household_key').size()
    past = tx[tx.day <= sday-28]
    mg = med_gap.reindex(pd.MultiIndex.from_frame(past[['household_key','product_id']].drop_duplicates()))
    due = (mg < 70).dropna()
    due = due[due].reset_index()[['household_key','product_id']].assign(_in=1)
    m = w1.merge(due, on=['household_key','product_id'], how='left')
    r = m[m._in==1].groupby('household_key').sales_value.sum()
    f['due70'] = (r/tot).where(tot>0)
    if 'rep12' in f: f['rep12_amt'] = f['rep12']*tot
    return f.reset_index()

rows=[]
for sday in TRAIN:
    f = habit_feats(tx_all, sday, med_gap); f['snapshot_day']=sday; rows.append(f)
H = pd.concat(rows, ignore_index=True)

base = agent_api.load_saved('e011_rank.parquet')
tt = agent_api.train_targets()
d = tt.merge(base.drop(columns=['household_key','snapshot_day']), left_index=True, right_index=True) \
      .merge(H, on=['household_key','snapshot_day'], how='left')

y = d.future_spend_4w.values
tr = (d.snapshot_day <= 403).values; va = (d.snapshot_day == 431).values
num = d.select_dtypes(include=[np.number]).drop(columns=['snapshot_day','future_spend_4w'])
d['s28_x_rep12'] = d.spend_28*d.rep12.fillna(0)
d['s28_x_due'] = d.spend_28*d.due70.fillna(0)
num2 = d.select_dtypes(include=[np.number]).drop(columns=['snapshot_day','future_spend_4w'])

def fit_eval(cols, tag, alpha=10.0, ret=False):
    X = d[cols].astype(float)
    X = X.fillna(0).values
    mu = X[tr].mean(0); sd = X[tr].std(0)+1e-9
    Xz = (X-mu)/sd
    keep = sd > 1e-9
    Xz = Xz[:, keep]
    A = np.hstack([np.ones((len(Xz),1)), Xz])
    lam = alpha/len(A[tr])
    I = np.eye(A.shape[1]); I[0,0]=0
    w = np.linalg.solve(A[tr].T@A[tr] + lam*I, A[tr].T@y[tr])
    p = A[va]@w
    mae = np.abs(p-y[va]).mean()
    print(tag, 'inner MAE %.2f (nfeat %d)' % (mae, keep.sum()))
    return (mae, p) if ret else mae

bcols = [c for c in num.columns if not c.startswith(('rep','jac','conc','nprod','due'))]
hcols = [c for c in num.columns if c.startswith(('rep','jac','conc','nprod','due'))]
fit_eval(bcols, 'base only')
fit_eval(bcols+hcols, 'base+habit')
fit_eval(hcols, 'habit only')
num2 = d.select_dtypes(include=[np.number]).drop(columns=['snapshot_day','future_spend_4w'])
b2 = [c for c in num2.columns if not c.startswith(('rep','jac','conc','nprod','due'))]
fit_eval(b2, 'base + s28 interactions')
fit_eval(b2+hcols, 'base + interactions + habit')
print('habit cols:', hcols)
# also check val-day target scale shift: mean y by snapshot
print(d.groupby('snapshot_day').future_spend_4w.mean().round(1).to_dict())


# ---- cell ----

import agent_api, pandas as pd, numpy as np
# rebuild d quickly (cache-free but cheap ops only) - reuse previous run's logic minimally
tt = agent_api.train_targets()
base = agent_api.load_saved('e011_rank.parquet')
print('base index name:', base.index.name, base.index[:3])
print('tt head:'); print(tt.head(3))
# check alignment: does base row i correspond to tt row i?
b = base.reset_index(drop=True)
m = tt.merge(b, left_index=True, right_index=True)
y = m.future_spend_4w.values
print('corr(spend_28, y) =', np.corrcoef(m.spend_28.fillna(0), y)[0,1])
print('MAE pred=spend_28 on day431:', np.abs(m.spend_28[m.snapshot_day==431].fillna(0).values - y[m.snapshot_day==431]).mean())
print('MAE pred=0.9*spend_28:', np.abs(0.9*m.spend_28[m.snapshot_day==431].fillna(0).values - y[m.snapshot_day==431]).mean())
# tiny ridge on just spend_28
tr = (m.snapshot_day<=403).values; va=(m.snapshot_day==431).values
X = m.spend_28.fillna(0).values
mu,sd = X[tr].mean(), X[tr].std()
Xz = ((X-mu)/sd).reshape(-1,1)
A = np.hstack([np.ones((len(Xz),1)), Xz])
w = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
print('ridge spend28 only coefs', w, 'MAE', np.abs(A[va]@w - y[va]).mean())
# now base numeric only, check per-column NaN rates and std
num = m.select_dtypes(include=[np.number]).drop(columns=['snapshot_day','future_spend_4w'])
print('n num cols', len(num))
nn = num.isna().mean().sort_values(ascending=False)
print('top NaN cols:'); print(nn.head(8))
print('cols with zero std:', (num.fillna(0).std()==0).sum())


# ---- cell ----

import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
base = agent_api.load_saved('e011_rank.parquet')
print('same household order:', (base.household_key.values == tt.household_key.values).mean())
print('same snapshot order:', (base.snapshot_day.values == tt.snapshot_day.values).mean())
print(base[['household_key','snapshot_day','spend_28']].head(4))
print(tt.head(4))
# proper merge on keys
m = tt.merge(base, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values
print('corr(spend_28,y) after key merge:', np.corrcoef(m.spend_28.fillna(0), y)[0,1])
va = m.snapshot_day==431
print('MAE spend28 on 431:', np.abs(m.spend_28[va].fillna(0).values - y[va]).mean())


# ---- cell ----

import agent_api, pandas as pd, numpy as np

tx_all = agent_api.snapshot().transactions
TRAIN = [95,123,151,179,207,235,263,291,319,347,375,403,431]
txs = tx_all.sort_values(['household_key','product_id','day'])
dif = txs.groupby(['household_key','product_id'], sort=False).day.diff()
med_gap = dif.groupby([txs.household_key, txs.product_id]).median()

def habit_feats(tx, sday, med_gap):
    w1 = tx[(tx.day > sday-28) & (tx.day <= sday)]
    w2 = tx[(tx.day > sday-56) & (tx.day <= sday-28)]
    w3 = tx[(tx.day > sday-84) & (tx.day <= sday-56)]
    w4 = tx[(tx.day > sday-112) & (tx.day <= sday-84)]
    tot = w1.groupby('household_key').sales_value.sum()
    f = pd.DataFrame(index=tot.index)
    def repshare(win, proddf, name):
        if len(win)==0: return
        p = proddf[['household_key','product_id']].drop_duplicates().assign(_in=1)
        m = win.merge(p, on=['household_key','product_id'], how='left')
        r = m[m._in==1].groupby('household_key').sales_value.sum()
        f[name] = (r/tot).where(tot>0)
    p1 = w1[['household_key','product_id']].drop_duplicates().assign(_a=1)
    if len(w2):
        p2 = w2[['household_key','product_id']].drop_duplicates().assign(_in=1)
        repshare(w1, p2, 'rep12')
        u = p1.merge(p2, on=['household_key','product_id'], how='outer', indicator=True)
        f['jacc12'] = u[u._merge=='both'].groupby('household_key').size()/u.groupby('household_key').size()
        if len(w3):
            p3 = w3[['household_key','product_id']].drop_duplicates().assign(_in=1)
            repshare(w1, p3, 'rep13')
            p23 = w2.merge(p3, on=['household_key','product_id'])[['household_key','product_id']].drop_duplicates().assign(_in=1)
            repshare(w1, p23, 'rep123')
    if len(w4):
        p4 = w4[['household_key','product_id']].drop_duplicates().assign(_in=1)
        wa = tx[(tx.day > sday-56) & (tx.day <= sday)]
        repshare(wa, p4, 'rep56')
    gp = w1.groupby(['household_key','product_id']).sales_value.sum().reset_index()
    gp['rk'] = gp.groupby('household_key').sales_value.rank(ascending=False, method='first')
    f['conc5'] = gp[gp.rk<=5].groupby('household_key').sales_value.sum()/tot
    f['nprod1'] = gp.groupby('household_key').size()
    if len(w2): f['nprod2'] = w2[['household_key','product_id']].drop_duplicates().groupby('household_key').size()
    if len(w3): f['nprod3'] = w3[['household_key','product_id']].drop_duplicates().groupby('household_key').size()
    past = tx[tx.day <= sday-28]
    mg = med_gap.reindex(pd.MultiIndex.from_frame(past[['household_key','product_id']].drop_duplicates()))
    due = (mg < 70).dropna()
    due = due[due].reset_index()[['household_key','product_id']].assign(_in=1)
    m = w1.merge(due, on=['household_key','product_id'], how='left')
    r = m[m._in==1].groupby('household_key').sales_value.sum()
    f['due70'] = (r/tot).where(tot>0)
    if 'rep12' in f: f['rep12_amt'] = f['rep12']*tot
    return f.reset_index()

rows=[]
for sday in TRAIN:
    f = habit_feats(tx_all, sday, med_gap); f['snapshot_day']=sday; rows.append(f)
H = pd.concat(rows, ignore_index=True)

tt = agent_api.train_targets()
base = agent_api.load_saved('e011_rank.parquet')
d = tt.merge(base, on=['household_key','snapshot_day'], how='left').merge(H, on=['household_key','snapshot_day'], how='left')
print('merged', d.shape, 'NaN spend28:', d.spend_28.isna().mean().round(3))
y = d.future_spend_4w.values
tr = (d.snapshot_day <= 403).values; va = (d.snapshot_day == 431).values
num = d.select_dtypes(include=[np.number]).drop(columns=['snapshot_day','future_spend_4w'])
d['s28_x_rep12'] = d.spend_28*d.rep12.fillna(0)
d['s28_x_due'] = d.spend_28*d.due70.fillna(0)
num2 = d.select_dtypes(include=[np.number]).drop(columns=['snapshot_day','future_spend_4w'])

def fit_eval(cols, tag, alpha=10.0):
    X = d[cols].astype(float).fillna(0).values
    mu = X[tr].mean(0); sd = X[tr].std(0)+1e-9
    keep = sd > 1e-9
    Xz = (X-mu)/sd[:,None if keep.ndim else 1] if False else (X-mu)/sd
    Xz = Xz[:, keep]
    A = np.hstack([np.ones((len(Xz),1)), Xz])
    w = np.linalg.solve(A[tr].T@A[tr] + (alpha/len(A[tr]))*np.eye(A.shape[1]), A[tr].T@y[tr])
    p = A[va]@w
    mae = np.abs(p-y[va]).mean()
    print(tag, 'inner MAE %.2f (nfeat %d)' % (mae, keep.sum()))
    return mae

bcols = [c for c in num.columns if not c.startswith(('rep','jac','conc','nprod','due'))]
hcols = [c for c in num.columns if c.startswith(('rep','jac','conc','nprod','due'))]
fit_eval(bcols, 'base only')
fit_eval(bcols+hcols, 'base+habit')
fit_eval(hcols, 'habit only')
num2 = d.select_dtypes(include=[np.number]).drop(columns=['snapshot_day','future_spend_4w'])
b2 = [c for c in num2.columns if not c.startswith(('rep','jac','conc','nprod','due'))]
fit_eval(b2, 'base + s28 interactions')
fit_eval(b2+hcols, 'base + interactions + habit')


# ---- cell ----

import agent_api, pandas as pd, numpy as np

tx_all = agent_api.snapshot().transactions
sd = 431
t = tx_all[tx_all.day <= sd]
hh = t.groupby('household_key').day.max().index
last_day = t.groupby('household_key').day.max().reindex(hh).values

def block_agg(k, col, agg):
    d1, d2 = sd-28*k-28, sd-28*k
    m = (t.day > d1) & (t.day <= d2)
    g = t[m].groupby('household_key')
    if agg=='sum': return g.sales_value.sum().reindex(hh).fillna(0).values
    return g.basket_id.nunique().reindex(hh).fillna(0).values

K = 13
BS = {k: block_agg(k,'sales_value','sum') for k in range(K+1)}
BB = {k: block_agg(k,'basket_id','n') for k in range(K+1)}
REC = {k: (sd-28*k - last_day).clip(0) for k in range(K)}

def design(k):
    s1,s2,s3,s4 = BS[k+1],BS[k+2],BS[k+3],BS[k+4]
    return np.column_stack([s1, s2, s3, s4, BB[k+1], REC[k], s1/(s1+s2+s3+1), (s1>0).astype(float)])

def ridge_fit(X, y, w, alpha):
    Xw = X*np.sqrt(w)[:,None]
    A = np.hstack([np.ones((len(X),1)), Xw])
    return np.linalg.solve(A.T@A + alpha*np.eye(A.shape[1]), A.T@(y*np.sqrt(w)))

X0 = design(0)
res = {}
for Kfit in [6, 9, 12]:
    for alpha in [1.0, 10.0, 100.0]:
        for decay in [1.0, 0.9]:
            Xs, ys, ws = [], [], []
            for k in range(1, Kfit+1):
                Xs.append(design(k)); ys.append(BS[k]); ws.append(np.full(len(hh), decay**(k-1)))
            wtr = ridge_fit(np.vstack(Xs), np.concatenate(ys), np.concatenate(ws), alpha)
            res[(Kfit,alpha,decay)] = np.hstack([np.ones((len(X0),1)), X0]) @ wtr

tt = agent_api.train_targets()
y431 = tt[tt.snapshot_day==431].set_index('household_key').future_spend_4w.reindex(hh).values
for key, pred in sorted(res.items()):
    print(key, 'MAE %.2f' % np.abs(pred - y431).mean())
print('s28 alone MAE %.2f' % np.abs(BS[1]-y431).mean())
print('0.9*s28 MAE %.2f' % np.abs(0.9*BS[1]-y431).mean())
best = min(res.items(), key=lambda kv: np.abs(kv[1]-y431).mean())
print('BEST:', best[0], 'MAE %.2f' % np.abs(best[1]-y431).mean())


# ---- cell ----

import agent_api, pandas as pd, numpy as np

tx_all = agent_api.snapshot().transactions
sd = 431
t = tx_all[tx_all.day <= sd]
last_day = t.groupby('household_key').day.max()
hh = last_day.index
last_day = last_day.values

def block_agg(k, agg):
    d1, d2 = sd-28*k-28, sd-28*k
    m = (t.day > d1) & (t.day <= d2)
    g = t[m].groupby('household_key')
    if agg=='sum': return g.sales_value.sum().reindex(hh).fillna(0).values
    return g.basket_id.nunique().reindex(hh).fillna(0).values

K = 16
BS = {k: block_agg(k,'sum') for k in range(K+1)}
BB = {k: block_agg(k,'n') for k in range(K+1)}
REC = {k: (sd-28*k - last_day).clip(0) for k in range(K)}

def design(k):
    s1,s2,s3,s4 = BS[k+1],BS[k+2],BS[k+3],BS[k+4]
    return np.column_stack([s1, s2, s3, s4, BB[k+1], REC[k], s1/(s1+s2+s3+1), (s1>0).astype(float)])

def ridge_fit(X, y, w, alpha):
    Xw = X*np.sqrt(w)[:,None]
    A = np.hstack([np.ones((len(X),1)), Xw])
    return np.linalg.solve(A.T@A + alpha*np.eye(A.shape[1]), A.T@(y*np.sqrt(w)))

X0 = design(0)
res = {}
for Kfit in [6, 9, 12]:
    for alpha in [1.0, 10.0, 100.0]:
        for decay in [1.0, 0.9]:
            Xs, ys, ws = [], [], []
            for k in range(1, Kfit+1):
                Xs.append(design(k)); ys.append(BS[k]); ws.append(np.full(len(hh), decay**(k-1)))
            wtr = ridge_fit(np.vstack(Xs), np.concatenate(ys), np.concatenate(ws), alpha)
            res[(Kfit,alpha,decay)] = np.hstack([np.ones((len(X0),1)), X0]) @ wtr

tt = agent_api.train_targets()
y431 = tt[tt.snapshot_day==431].set_index('household_key').future_spend_4w.reindex(hh).values
for key, pred in sorted(res.items()):
    print(key, 'MAE %.2f' % np.abs(pred - y431).mean())
print('s28 alone MAE %.2f' % np.abs(BS[1]-y431).mean())
print('0.9*s28 MAE %.2f' % np.abs(0.9*BS[1]-y431).mean())
best = min(res.items(), key=lambda kv: np.abs(kv[1]-y431).mean())
print('BEST:', best[0], 'MAE %.2f' % np.abs(best[1]-y431).mean())


# ---- cell ----

import agent_api, pandas as pd, numpy as np

def make_fn():
    def fn(view, sday):
        tx = view.transactions
        g = tx.groupby('household_key').day.max()
        hh_all = g.index
        last = g.values.astype(float)
        K = 16
        BS = {}; BB = {}
        for k in range(K+1):
            d1, d2 = sday-28*k-28, sday-28*k
            m = (tx.day > d1) & (tx.day <= d2)
            sub = tx[m]
            BS[k] = sub.groupby('household_key').sales_value.sum().reindex(hh_all).fillna(0).values
            BB[k] = sub.groupby('household_key').basket_id.nunique().reindex(hh_all).fillna(0).values
        REC = {k: np.clip(sday-28*k - last, 0, None) for k in range(K)}
        def design(k):
            s1,s2,s3,s4 = BS[k+1],BS[k+2],BS[k+3],BS[k+4]
            return np.column_stack([s1,s2,s3,s4,BB[k+1],REC[k],s1/(s1+s2+s3+1),(s1>0).astype(float)])
        def design_sp(k):
            s1,s2,s3,s4 = BS[k+1],BS[k+2],BS[k+3],BS[k+4]
            return np.column_stack([s1,s2,s3,s4])
        def ridge_fit(X, y, w, alpha):
            Xw = X*np.sqrt(w)[:,None]
            A = np.hstack([np.ones((len(X),1)), Xw])
            return np.linalg.solve(A.T@A + alpha*np.eye(A.shape[1]), A.T@(y*np.sqrt(w)))
        X0 = design(0); X0sp = design_sp(0)
        out = pd.DataFrame(index=hh_all)
        for Kfit, tag in [(9,'9'),(6,'6'),(12,'12')]:
            Xs=[];ys=[];ws=[]
            for k in range(1,Kfit+1):
                Xs.append(design(k)); ys.append(BS[k]); ws.append(np.full(len(hh_all), 0.9**(k-1)))
            wtr = ridge_fit(np.vstack(Xs), np.concatenate(ys), np.concatenate(ws), 1.0)
            out['emb_pred_'+tag] = np.hstack([np.ones((len(X0),1)), X0]) @ wtr
        Xs=[];ys=[];ws=[]
        for k in range(1,10):
            Xs.append(design_sp(k)); ys.append(BS[k]); ws.append(np.full(len(hh_all), 0.9**(k-1)))
        wtr = ridge_fit(np.vstack(Xs), np.concatenate(ys), np.concatenate(ws), 1.0)
        out['emb_sp'] = np.hstack([np.ones((len(X0sp),1)), X0sp]) @ wtr
        out['emb_clip'] = np.clip(out['emb_pred_9'].values, 0, None)
        out['emb_ratio'] = out['emb_pred_9'].values/(BS[1]+1)
        out['emb_diff'] = out['emb_pred_9'].values - BS[1]
        need = pd.Index(view.households) if not hasattr(view.households, 'columns') else pd.Index(view.households.household_key)
        out = out.reindex(need)
        for c in out.columns: out[c] = out[c].fillna(0)
        return out
    return fn

F = agent_api.build_features(make_fn())
print('F:', F.shape, list(F.columns)[:12])
print(F.head(3))
base = agent_api.load_saved('e011_rank.parquet')
print('base:', base.shape)
if 'snapshot_day' in F.columns:
    M = base.merge(F, on=['household_key','snapshot_day'], how='left')
else:
    F2 = F.reset_index().rename(columns={'index':'household_key'})
    M = base.merge(F2, on=['household_key','snapshot_day'], how='left')
print('merged:', M.shape, 'NaN emb:', M[[c for c in M.columns if c.startswith('emb')]].isna().mean().round(3).to_dict())
path = agent_api.save_table(M, 'e012_embed.parquet')
print('saved:', path)
