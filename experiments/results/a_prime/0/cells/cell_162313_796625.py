import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
tt = agent_api.train_targets()
base = agent_api.load_saved("e013_stock.parquet")
keys=["household_key","snapshot_day"]
df = tt.merge(base,on=keys,how="left")
cols0=[c for c in base.columns if c not in keys]
for c in cols0:
    if df[c].dtype=='object' or str(df[c].dtype)=='category':
        df[c]=df[c].astype('object').astype('category').cat.codes.astype(float).replace(-1,np.nan)

def ridge_eval(frame, feat_cols, tag, alphas=(8,16,32,64,128)):
    fit=frame[frame.snapshot_day<=403]; hold=frame[frame.snapshot_day==431]
    yf=fit.future_spend_4w.values; yh=hold.future_spend_4w.values
    Xf=fit[feat_cols].astype(float).values; Xh=hold[feat_cols].astype(float).values
    med=np.nanmedian(Xf,axis=0); Xf=np.where(np.isnan(Xf),med,Xf); Xh=np.where(np.isnan(Xh),med,Xh)
    mu=Xf.mean(0); sd=Xf.std(0)+1e-9; Xf=(Xf-mu)/sd; Xh=(Xh-mu)/sd
    Xf=np.hstack([Xf,np.ones((len(Xf),1))]); Xh=np.hstack([Xh,np.ones((len(Xh),1))])
    itr=(fit.snapshot_day<=375).values; iva=(fit.snapshot_day==403).values
    best=None
    for a in alphas:
        A=Xf[itr]; w=np.linalg.solve(A.T@A+a*np.eye(A.shape[1]),A.T@yf[itr])
        mae=np.abs(np.clip(Xf[iva]@w,0,None)-yf[iva]).mean()
        if best is None or mae<best[1]: best=(a,mae)
    a=best[0]; G=Xf.T@Xf+a*np.eye(Xf.shape[1]); w=np.linalg.solve(G,Xf.T@yf)
    ph=np.clip(Xh@w,0,None); mae=np.abs(ph-yh).mean()
    print(f"[{tag}] alpha={a} inner={best[1]:.2f} hold431={mae:.3f}")
    return mae

lvl = df['x_exp4w'].fillna(0).values
rec = df['recency'].fillna(999).values
s84 = df['spend_84'].fillna(0).values
s28 = df['spend_28'].fillna(0).values
b28 = df['baskets_28'].fillna(0).values

cands = {}
cands['sqrt'] = pd.DataFrame({'f_sqrt_exp': np.sqrt(np.clip(lvl,0,None)), 'f_sqrt_s84': np.sqrt(s84)})
g10 = np.exp(-rec/10.0); g7 = np.exp(-rec/7.0)
cands['decaygate'] = pd.DataFrame({'f_g10': g10, 'f_exp*g10': lvl*g10, 'f_s84*g10': s84*g10, 'f_g7': g7, 'f_exp*g7': lvl*g7})
inact = (rec>14).astype(float)
cands['inactgate'] = pd.DataFrame({'f_inact': inact, 'f_exp*inact': lvl*inact, 'f_s84*inact': s84*inact, 'f_exp*(1-inact)': lvl*(1-inact)})
# long-dormancy: no purchase in 28d
inact28 = (rec>28).astype(float)
cands['inact28'] = pd.DataFrame({'f_inact28': inact28, 'f_exp*inact28': lvl*inact28, 'f_s84*inact28': s84*inact28})
# trend-based gate: declining blocks
trend = df['trend_28'].fillna(1).values
cands['trendgate'] = pd.DataFrame({'f_tr': trend, 'f_exp*tr': lvl*np.clip(trend,0,3)})

m_ref = ridge_eval(df, cols0, "ref e013")
for k,v in cands.items():
    d2 = pd.concat([df, v], axis=1)
    ridge_eval(d2, cols0+list(v.columns), "e013+"+k)
# each candidate ALONE replacing nothing, also try gate features only with core levels
core = [c for c in cols0 if c.startswith(('x_','spend_','baskets_','recency','active','days_','trend','basket_val'))]
d2 = pd.concat([df, cands['decaygate'], cands['inactgate']], axis=1)
ridge_eval(d2, cols0+list(cands['decaygate'].columns)+list(cands['inactgate'].columns), "e013+decay+inact")
