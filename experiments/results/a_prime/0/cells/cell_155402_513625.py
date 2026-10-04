
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
