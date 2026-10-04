import agent_api, pandas as pd, numpy as np

def ridge_fit_pred(Xtr, ytr, Xva, alpha):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Z = (Xtr-mu)/sd; Zv = (Xva-mu)/sd
    Z = np.clip(np.c_[np.ones(len(Z)), Z], -50, 50)
    Zv = np.clip(np.c_[np.ones(len(Zv)), Zv], -50, 50)
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[-1,-1]-=alpha
    w = np.linalg.solve(A, Z.T@ytr)
    return np.clip(Zv@w, 0, None)

keys=['household_key','snapshot_day']
df = agent_api.load_saved('e009_demo.parquet').merge(agent_api.train_targets(), on=keys, how='left')
fcols=[c for c in df.columns if c not in keys+['future_spend_4w']]
d=df.snapshot_day.values; tr_days=agent_api.snapshot_days()['train']

def get_X(df, mode):
    X = df[fcols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
    if mode=='base': return X
    extra=[]
    # hinges on top spend features at percentiles 25/50/75 of train dist
    top=['spend_84','spend_28','fwd28_mean','wk_avg_84','spend_lag1','ewma28_4w','x_life_rate_wk']
    for c in top:
        v = df[c].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
        for q in (0.25,0.5,0.75):
            t = np.quantile(v, q)
            extra.append(np.maximum(v-t,0))
    if mode=='hinge':
        return np.c_[X, np.array(extra).T]
    if mode=='rank':
        # rank-transform top 20 heavy-tailed features
        rtop=['spend_84','spend_28','fwd28_mean','wk_avg_84','spend_lag1','ewma28_4w','x_life_rate_wk',
              'spend_364','spend_life','x_life_total','spend_56','fwd28_k1','fwd28_median','wk_avg_8',
              'spend_112','spend_168','x_s28','fwd28_max','spend_7','x_ya4w']
        R = df[rtop].apply(pd.to_numeric, errors='coerce').fillna(0).rank(pct=True).values
        return np.c_[X, R]
    if mode=='hinge+rank':
        R = df[['spend_84','fwd28_mean']].apply(pd.to_numeric, errors='coerce').fillna(0).rank(pct=True).values
        return np.c_[X, np.array(extra).T, R]

for mode in ['base','hinge','rank','hinge+rank']:
    mas=[]
    for inner_day in [403,431]:
        trin=df[np.isin(d,[x for x in tr_days if x<inner_day])]; inner=df[d==inner_day]
        Xtr,Xin = get_X(trin,mode), get_X(inner,mode)
        p = ridge_fit_pred(Xtr, trin.future_spend_4w.values, Xin, 3000.)
        mas.append(np.abs(p-inner.future_spend_4w.values).mean())
    print(f'{mode:12s} d403:{mas[0]:.3f} d431:{mas[1]:.3f} avg:{np.mean(mas):.3f}')
