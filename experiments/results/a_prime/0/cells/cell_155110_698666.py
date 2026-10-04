
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
