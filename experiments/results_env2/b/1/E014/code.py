import agent_api, pandas as pd, numpy as np
for name in ["candA_dm","candB_g","candC_hb","candD_su","candE_td","candGST"]:
    try:
        df = agent_api.load_saved(name+".parquet")
        print(name, df.shape)
        print(list(df.columns)[:40])
        print("---")
    except Exception as e:
        print(name, "ERR", type(e).__name__, e)


# ---- cell ----
import agent_api, pandas as pd, numpy as np
for name in ["candA_dm","candB_g","candC_hb","candD_su","candE_td","candGST"]:
    df = agent_api.load_saved(name+".parquet")
    cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
    print(name, len(cols))
    print(cols[60:])
    print("===")


# ---- cell ----
import agent_api, pandas as pd
for name in ["candC_hb","candD_su"]:
    df = agent_api.load_saved(name+".parquet")
    cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
    print(name, len(cols))
    print(cols[120:])
    print("===")


# ---- cell ----
import agent_api, pandas as pd, numpy as np

e8 = agent_api.load_saved("e008_level_shape.parquet")
tt = agent_api.train_targets()
print("e8 shape", e8.shape, "tt shape", tt.shape)
print(tt['future_spend_4w'].describe())
print("zero share:", (tt['future_spend_4w']==0).mean())
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median',lambda s:(s==0).mean()]))

tr = tt.merge(e8, on=["household_key","snapshot_day"], how="left")
print("merged", tr.shape)
y = tr['future_spend_4w'].values

def mae(p): return np.mean(np.abs(p-y))

cands = {
 "sp28": np.expm1(tr['z_log_sp28'].fillna(0).values),
 "sp56h": np.expm1(tr['z_log_sp56'].fillna(0).values)/2,
 "sp84h": np.expm1(tr['z_log_sp84'].fillna(0).values)/3,
 "med4w": tr['z_med4w_hist'].fillna(0).values,
 "meanwk4": tr['z_mean_week_spend_all'].fillna(0).values*4,
}
for k,v in cands.items():
    print(k, "mae=%.2f corr=%.3f" % (mae(v), np.corrcoef(v,y)[0,1]))

# blend optimization on early train snapshots, eval on later
w_days_fit = [95,123,151,179,207,235]
w_days_ev  = [263,291,319,347,375,403,431]
fit = tr['snapshot_day'].isin(w_days_fit).values
ev  = tr['snapshot_day'].isin(w_days_ev).values
keys = list(cands.keys())
X = np.column_stack([cands[k] for k in keys])
from itertools import product
best=None
for w in product(np.linspace(0,1,11), repeat=len(keys)):
    if abs(sum(w)-1)>0.01: continue
    p = X[fit]@np.array(w)
    m = np.mean(np.abs(p-y[fit]))
    if best is None or m<best[0]: best=(m,w)
print("best blend fit:", best)
w=np.array(best[1]); p=X[ev]@w
print("blend ev mae=%.2f (weights %s)" % (np.mean(np.abs(p-y[ev])), dict(zip(keys,w.round(2)))))
print("single best on ev:", {k: round(mae(X[ev][:,i]),2) for i,k in enumerate(keys)})


# ---- cell ----
import agent_api, pandas as pd, numpy as np

e8 = agent_api.load_saved("e008_level_shape.parquet")
tt = agent_api.train_targets()
tr = tt.merge(e8, on=["household_key","snapshot_day"], how="left")
y = tr['future_spend_4w'].values

# check dtypes
num_cols = [c for c in e8.columns if c not in ("household_key","snapshot_day") and pd.api.types.is_numeric_dtype(e8[c])]
cat_cols = [c for c in e8.columns if c not in ("household_key","snapshot_day") and not pd.api.types.is_numeric_dtype(e8[c])]
print("numeric:", len(num_cols), "categorical:", len(cat_cols), cat_cols)

# scales of key features
key = ['sp28','sp56','sp84','sp168','sp364','z_log_sp28','z_med4w_hist','z_mean_week_spend_all','days_since_last','tenure','wksp_mean','splag1y']
print(e8[key].describe().T[['mean','std','min','max']])

# global median predictor MAE on all train rows
print("global median mae:", np.mean(np.abs(y-np.median(y))))
# corr of key feats with target
for c in key:
    print(c, round(np.corrcoef(tr[c].fillna(0), y)[0,1],3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

tt = agent_api.train_targets()

def prep(df):
    df = tt.merge(df, on=["household_key","snapshot_day"], how="inner")
    y = df['future_spend_4w'].values.astype(float)
    cats = [c for c in df.columns if c not in ("household_key","snapshot_day","future_spend_4w") and not pd.api.types.is_numeric_dtype(df[c])]
    nums = [c for c in df.columns if c not in ("household_key","snapshot_day","future_spend_4w") and pd.api.types.is_numeric_dtype(df[c])]
    Xn = df[nums].astype(float).fillna(0).values
    Xc = pd.get_dummies(df[cats].astype(str), dummy_na=True).values.astype(float) if cats else np.zeros((len(df),0))
    X = np.column_stack([Xn, Xc]) if Xc.size else Xn
    return X, y, df['snapshot_day'].values

def ridge_fit(X, y, alpha):
    mu = X.mean(0); sd = X.std(0); sd[sd==0]=1
    Z = (X-mu)/sd
    A = Z.T@Z + alpha*np.eye(Z.shape[1])
    b = Z.T@y
    w = np.linalg.solve(A, b)
    return w, mu, sd

def ridge_pred(X, model):
    w, mu, sd = model
    return (X-mu)/sd @ w

def local_eval(path, fit_days, ev_days, alpha=100.0):
    df = agent_api.load_saved(path)
    X, y, day = prep(df)
    f = np.isin(day, fit_days); e = np.isin(day, ev_days)
    m = ridge_fit(X[f], y[f], alpha)
    p = ridge_pred(X[e], m)
    return np.mean(np.abs(p - y[e]))

print("alpha check on e008 (fit 95-347, ev 375/403/431):")
for a in [1,10,30,100,300,1000]:
    print(a, round(local_eval("e008_level_shape.parquet", list(range(95,348,28)), [375,403,431], a),3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
df = agent_api.load_saved("candC_hb.parquet")
print(df.shape, "rows ok:", df['household_key'].nunique())
m = tt.merge(df, on=["household_key","snapshot_day"])
new = ['hb_repeat_share_28','hb_rep_prod_frac_28','hb_dept_entropy_84','hb_top1prod_share_84','hb_new_prod_share_28']
print(m[new].describe().T[['mean','std','min','max']])
y = m['future_spend_4w'].values
for c in new:
    print(c, "corr", round(np.corrcoef(m[c].fillna(0), y)[0,1],3))
# also check candAll composition
ca = agent_api.load_saved("candAll.parquet")
print("candAll", ca.shape)
print([c for c in ca.columns if c.startswith(('hb_','su_','td_','g_','dm_'))])


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
tt = agent_api.train_targets()

def prep(df):
    df = tt.merge(df, on=["household_key","snapshot_day"], how="inner")
    y = df['future_spend_4w'].values.astype(float)
    cats = [c for c in df.columns if c not in ("household_key","snapshot_day","future_spend_4w") and not pd.api.types.is_numeric_dtype(df[c])]
    nums = [c for c in df.columns if c not in ("household_key","snapshot_day","future_spend_4w") and pd.api.types.is_numeric_dtype(df[c])]
    Xn = df[nums].astype(float).fillna(0).values
    Xc = pd.get_dummies(df[cats].astype(str), dummy_na=True).values.astype(float) if cats else np.zeros((len(df),0))
    X = np.column_stack([Xn, Xc]) if Xc.size else Xn
    return X, y, df['snapshot_day'].values

def local_eval(path, fit_days, ev_days, alpha=300.0):
    df = agent_api.load_saved(path)
    X, y, day = prep(df)
    f = np.isin(day, fit_days); e = np.isin(day, ev_days)
    mu = X[f].mean(0); sd = X[f].std(0); sd[sd==0]=1
    Z=(X-mu)/sd
    A=Z[f].T@Z[f]+alpha*np.eye(Z.shape[1]); b=Z[f].T@y[f]
    w=np.linalg.solve(A,b)
    p=Z[e]@w
    return np.mean(np.abs(p-y[e]))

fit=list(range(95,348,28)); ev=[375,403,431]
print("local ridge probe (fit 95-347, ev 375-431):")
for name in ["e008_level_shape","e013_union","candAll","candA_dm","candB_g","candC_hb","candD_su","candE_td","candGST"]:
    try:
        print(name, round(local_eval(name+".parquet", fit, ev),3))
    except Exception as e:
        print(name,"ERR",type(e).__name__,e)


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
tt = agent_api.train_targets()

def prep(df):
    df = tt.merge(df, on=["household_key","snapshot_day"], how="inner")
    y = df['future_spend_4w'].values.astype(float)
    cats = [c for c in df.columns if c not in ("household_key","snapshot_day","future_sp4w_dummy") and not pd.api.types.is_numeric_dtype(df[c])]
    cats = [c for c in df.columns if c not in ("household_key","snapshot_day","future_spend_4w") and not pd.api.types.is_numeric_dtype(df[c])]
    nums = [c for c in df.columns if c not in ("household_key","snapshot_day","future_spend_4w") and pd.api.types.is_numeric_dtype(df[c])]
    Xn = df[nums].astype(float).fillna(0).values
    Xc = pd.get_dummies(df[cats].astype(str), dummy_na=True).values.astype(float) if cats else np.zeros((len(df),40))
    X = np.column_stack([Xn, Xc]) if Xc.size else Xn
    return X, y, df['snapshot_day'].values

def local_eval(path, fit_days, ev_days, alpha=300.0):
    df = agent_api.load_saved(path)
    X, y, day = prep(df)
    f = np.isin(day, fit_days); e = np.isin(day, ev_days)
    mu = X[f].mean(0); sd = X[f].std(0); sd[sd==0]=1
    Z=(X-mu)/sd
    A=Z[f].T@Z[f]+alpha*np.eye(Z.shape[1]); b=Z[f].T@y[f]
    w=np.linalg.solve(A,b)
    p=Z[e]@w
    return np.mean(np.abs(p-y[e]))

# try larger alphas; ridge underfitting suggests alpha too small or features too noisy
fit=list(range(95,348,28)); ev=[375,403,431]
for a in [300, 1000, 3000, 10000, 30000]:
    print("e008 alpha",a, round(local_eval("e008_level_shape.parquet", fit, ev, a),3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
tt = agent_api.train_targets()

def prep(df):
    df = tt.merge(df, on=["household_key","snapshot_day"], how="inner")
    y = df['future_spend_4w'].irreducible
    return df

def ridge_on_cols(df, cols, fit_days, ev_days, alpha):
    d = tt.merge(df[["household_key","snapshot_day"]+cols], on=["household_key","snapshot_day"], how="inner")
    y = d['future_spend_4w'].values.astype(float)
    X = d[cols].astype(float).fillna(0).values
    f = np.isin(d['snapshot_day'], fit_days); e = np.isin(d['snapshot_day'], ev_days)
    mu=X[f].mean(0); sd=X[f].std(0); sd[sd==0]=1
    Z=(X-mu)/sd
    A=Z[f].T@Z[f]+alpha*np.eye(len(cols)); b=Z[f].T@y[f]
    w=np.linalg.solve(A,b)
    p=Z[e]@w
    return np.mean(np.abs(p-y[e]))

e8 = agent_api.load_saved("e008_level_shape.parquet")
fit=list(range(95,348,28)); ev=[375,403,431]
num = [c for c in e8.columns if c not in ("household_key","snapshot_day") and pd.api.types.is_numeric_dtype(e8[c])]

# single-feature probes at big alpha
res=[]
for c in num:
    try:
        m = ridge_on_cols(e8, [c], fit, ev, 30000)
        res.append((m,c))
    except Exception as ex:
        pass
res.sort()
print("best single numeric features (local ridge):")
for m,c in res[:15]: print(round(m,2), c)
print("...")
for m,c in res[-5:]: print(round(m,2), c)


# ---- cell ----
import agent_api, pandas as long_d


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
tt = agent_api.train_targets()
e8 = agent_api.load_saved("e008_level_shape.parquet")

def ridge_on_cols(df, cols, fit_days, ev_days, alpha):
    d = tt.merge(df[["household_key","snapshot_day"]+cols], on=["household_key","snapshot_day"], how="inner")
    y = d['future_spend_4w'].values.astype(float)
    X = d[cols].astype(float).fillna(0).values
    f = np.isin(d['snapshot_day'], fit_days); e = np.isin(d['snapshot_day'], ev_days)
    mu=X[f].mean(0); sd=X[f].std(0); sd[sd==0]=1
    Z=(X-mu)/sd
    A=Z[f].T@Z[f]+alpha*np.eye(len(cols)); b=Z[f].T@y[f]
    w=np.linalg.solve(A,b)
    p=Z[e]@w
    return np.mean(np.abs(p-y[e]))

fit=list(range(95,348,28)); ev=[375,403,431]
# small curated sets
sets = {
 "sp728 only": ["sp728"],
 "sp364+728": ["sp364","sp728"],
 "top5": ["sp728","sp364","sp364_rate","prods728","nact728"],
 "top10": ["sp728","sp364","sp364_rate","prods728","nact728","rdisc364","nact364","trips728","trips364","z_max4w_hist"],
 "sp84+sp728": ["sp84","sp728"],
 "sp84+sp728+recency": ["sp84","sp728","days_since_last"],
 "sp84+sp728+med4w": ["sp84","sp728","z_med4w_hist"],
 "sp28+84+728+med4w": ["sp28","sp84","sp728","z_med4w_hist"],
}
for k,v in sets.items():
    print(k, round(ridge_on_cols(e8, v, fit, ev, 30000),3))
