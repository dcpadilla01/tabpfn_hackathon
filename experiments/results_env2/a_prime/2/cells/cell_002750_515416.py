import pandas as pd, numpy as np, agent_api

def prep(df):
    tt = agent_api.train_targets()
    m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
    y = m['future_spend_4w'].astype(float).values
    sd = m['snapshot_day'].values
    X = pd.get_dummies(m.drop(columns=['future_spend_4w']), dummy_na=True).astype(float)
    X = X.replace([np.inf,-np.inf], np.nan).values
    return X, y, sd

def fit_eval(X, y, sd, vd, a, clip0):
    tr = sd < vd; va = sd == vd
    mu = np.nanmean(X[tr],0); sg = np.nanstd(X[tr],0)+1e-9
    mu = np.where(np.isnan(mu),0.,mu); sg = np.where(np.isnan(sg),1.,sg)
    Z = np.where(np.isnan(X[tr]),mu,X[tr]); Z=(Z-mu)/sg
    Zv = np.where(np.isnan(X[va]),mu,X[va]); Zv=(Zv-mu)/sg
    if a == 0:
        w = np.linalg.lstsq(Z, y[tr]-y[tr].mean(), rcond=None)[0]
    else:
        w = np.linalg.solve(Z.T@Z + a*np.eye(Z.shape[1]), Z.T@(y[tr]-y[tr].mean()))
    p = Zv@w + y[tr].mean()
    if clip0: p = np.clip(p, 0, None)
    return np.abs(p - y[va]).mean()

def cv(path, a, clip0, vds=(403,431)):
    df = agent_api.baseline_features() if path=='BASELINE' else agent_api.load_saved(path)
    X,y,sd = prep(df)
    return np.mean([fit_eval(X,y,sd,vd,a,clip0) for vd in vds])

paths = [('E000','BASELINE'),('E001','e001_recent_behavior.parquet'),('E005','e005_decay_gapcv.parquet'),
         ('E010','e010_lifecycle.parquet'),('E011','e011_discounts.parquet'),
         ('E012','e012_hh_target_enc.parquet'),('E015','e015_best_pseudo.parquet')]
harness = {'E000':92.45,'E001':63.57,'E005':63.32,'E010':62.70,'E011':62.65,'E012':72.18,'E015':62.65}

for a, clip0 in [(0,False),(1,False),(10,False),(1,True),(10,True)]:
    res = {n: cv(p,a,clip0) for n,p in paths}
    ns = list(res); 
    r1 = np.argsort(np.argsort([res[n] for n in ns])); r2 = np.argsort(np.argsort([harness[n] for n in ns]))
    corr = np.corrcoef(r1, r2)[0,1]
    print(f'a={a} clip0={clip0}: ' + ' '.join(f'{n}={res[n]:.2f}' for n in ns) + f'  rankcorr={corr:.2f}')
