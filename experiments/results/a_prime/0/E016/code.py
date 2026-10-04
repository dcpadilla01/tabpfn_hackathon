import agent_api, pandas as pd, numpy as np
for name in ["e013_stock.parquet","e011_rank.parquet","stock_v1.parquet","rhythm_v1.parquet","mkt_v2.parquet","e012_embed.parquet"]:
    df = agent_api.load_saved(name)
    print("==", name, df.shape)
    print(list(df.columns))
print()
v = agent_api.snapshot()
dm = v.display_mailer
print("display_mailer", dm.shape)
print(dm.head(8))
print("display uniq:", dm.display.unique()[:20])
print("mailer uniq:", dm.mailer.unique()[:20])
cr = v.coupon_redemptions
print("coupon_redemptions", cr.shape); print(cr.head())
cp = v.coupons
print("coupons", cp.shape); print(cp.head())
t = v.transactions
print(t[['sales_value','coupon_disc','coupon_match_disc','retail_disc','quantity','trans_time']].describe())
print(agent_api.snapshot_days())


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

tt = agent_api.train_targets()
y_all = tt.future_spend_4w
print("train rows", len(tt), "zero share %.3f" % (y_all==0).mean())
print(y_all.describe())
print("p50/75/90/95/99:", np.percentile(y_all,[50,75,90,95,99]).round(1))

for name in ["structure_v1.parquet","hist_v2.parquet","mix_v1.parquet","season_v1.parquet"]:
    df = agent_api.load_saved(name)
    print("==",name,df.shape); print(list(df.columns))

base = agent_api.load_saved("e013_stock.parquet")
keys=["household_key","snapshot_day"]
cols=[c for c in base.columns if c not in keys]
df = tt.merge(base,on=keys,how="left")
cat_cols=[c for c in cols if df[c].dtype=='object' or str(df[c].dtype)=='category']
print("cat cols:",cat_cols)
for c in cat_cols:
    df[c]=df[c].astype('object').astype('category')
    df[c]=df[c].cat.codes.astype(float).replace(-1,np.nan)

def ridge_eval(frame, feat_cols, tag, alphas=(0.5,1,2,4,8,16,32)):
    fit=frame[frame.snapshot_day<=403]; hold=frame[frame.snapshot_day==431]
    yf=fit.future_spend_4w.values; yh=hold.future_spend_4w.values
    Xf=fit[feat_cols].astype(float).values; Xh=hold[feat_cols].astype(float).values
    med=np.nanmedian(Xf,axis=0)
    Xf=np.where(np.isnan(Xf),med,Xf); Xh=np.where(np.isnan(Xh),med,Xh)
    mu=Xf.mean(0); sd=Xf.std(0)+1e-9
    Xf=(Xf-mu)/sd; Xh=(Xh-mu)/sd
    Xf=np.hstack([Xf,np.ones((len(Xf),1))]); Xh=np.hstack([Xh,np.ones((len(Xh),1))])
    itr=(fit.snapshot_day<=375).values; iva=(fit.snapshot_day==403).values
    best=None
    for a in alphas:
        A=Xf[itr]; G=A.T@A+a*np.eye(A.shape[1]); w=np.linalg.solve(G,A.T@yf[itr])
        mae=np.abs(Xf[iva]@w-yf[iva]).mean()
        if best is None or mae<best[1]: best=(a,mae)
    a=best[0]
    G=Xf.T@Xf+a*np.eye(Xf.shape[1]); w=np.linalg.solve(G,Xf.T@yf)
    ph=np.clip(Xh@w,0,None)
    mae=np.abs(ph-yh).mean()
    s28=hold['spend_28'].fillna(0).values
    print(f"[{tag}] alpha={a} innerMAE={best[1]:.2f} hold431 MAE={mae:.3f} | naive spend_28 MAE={np.abs(s28-yh).mean():.3f} corr={np.corrcoef(ph,yh)[0,1]:.3f}")
    return mae

m0=ridge_eval(df,cols,"base e013")
# correlation screen
fit=df[df.snapshot_day<=403]
cors={}
for c in cols:
    x=fit[c].astype(float).values; yy=fit.future_spend_4w.values
    ok=~np.isnan(x)
    if ok.sum()>100: cors[c]=np.corrcoef(x[ok],yy[ok])[0,1]
cs=sorted(cors.items(),key=lambda kv:-abs(kv[1]))
print("top |corr| with target:", [(k,round(v,3)) for k,v in cs[:25]])
print("bottom:", [(k,round(v,3)) for k,v in cs[-10:]])


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
tt = agent_api.train_targets()
base = agent_api.load_saved("e013_stock.parquet")
keys=["household_key","snapshot_day"]
df = tt.merge(base,on=keys,how="left")
cols=[c for c in base.columns if c not in keys]
for c in cols:
    if df[c].dtype=='object' or str(df[c].dtype)=='category':
        df[c]=df[c].astype('object').astype('category').cat.codes.astype(float).replace(-1,np.nan)
fit=df[df.snapshot_day<=403]; hold=df[df.snapshot_day==431]
def prep(Xf,Xh):
    med=np.nanmedian(Xf,axis=0); Xf=np.where(np.isnan(Xf),med,Xf); Xh=np.where(np.isnan(Xh),med,Xh)
    mu=Xf.mean(0); sd=Xf.std(0)+1e-9
    return (Xf-mu)/sd,(Xh-mu)/sd
Xf, Xh = prep(fit[cols].astype(float).values, hold[cols].astype(float).values)
Xf=np.hstack([Xf,np.ones((len(Xf),1))]); Xh=np.hstack([Xh,np.ones((len(Xh),1))])
yf=fit.future_spend_4w.values; yh=hold.future_spend_4w.values
A=Xf[fit.snapshot_day<=375]; w=np.linalg.solve(A.T@A+32*np.eye(A.shape[1]),A.T@yf[fit.snapshot_day<=375])
ph=np.clip(Xh@w,0,None)
res=ph-yh
yh0=yh==0
print("HOLD 431: n=%d  MAE=%.2f"%(len(yh),np.abs(res).mean()))
print("zeros: n=%d (%.0f%%)  mean|res|=%.1f  mean pred on zeros=%.1f"%(yh0.sum(),100*yh0.mean(),np.abs(res[yh0]).mean(),ph[yh0].mean()))
print("actives: mean|res|=%.1f"%np.abs(res[~yh0]).mean())
for lo,hi in [(0,50),(50,150),(150,300),(300,600),(600,3000)]:
    m=(yh>=lo)&(yh<hi)
    print(f"y in [{lo},{hi}): n={m.sum():4d} mean pred={ph[m].mean():7.1f} mean y={yh[m].mean():7.1f} MAE={np.abs(res[m]).mean():6.1f}")
# deciles of prediction
q=np.quantile(ph,[.5,.75,.9,.95,.99])
print("pred quantiles:",q.round(1))
print("y quantiles:",np.quantile(yh,[.5,.75,.9,.95,.99]).round(1))
# how much would perfect zero-classification help: set pred=0 where yh==0
print("MAE if zeros perfectly identified:", np.abs(np.where(yh0,0,ph)-yh).mean())
# error on train fit rows (in-sample-ish) by y bucket
ptr=np.clip(Xf@w,0,None); rtr=ptr-yf
for lo,hi in [(0,1),(1,50),(50,150),(150,300),(300,600),(600,3000)]:
    m=(yf>=lo)&(yf<hi)
    print(f"TRAIN y in [{lo},{hi}): n={m.sum():5d} mean pred={ptr[m].mean():7.1f} mean y={yf[m].mean():7.1f} MAE={np.abs(rtr[m]).mean():6.1f}")
print("TRAIN zeros: n=%d mean pred=%.1f MAE=%.1f"%((yf==0).sum(),ptr[yf==0].mean(),np.abs(rtr[yf==0]).mean()))


# ---- cell ----
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


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
v = agent_api.snapshot()  # capped at 459
t = v.transactions[['household_key','day','sales_value','basket_id']]
first = t.groupby('household_key').day.min().rename('first_day')

def win_spend(lo, hi):
    m = t[(t.day>=lo)&(t.day<=hi)]
    return m.groupby('household_key').sales_value.sum()

days = [95,123,151,179,207,235,263,291,319,347,375,403,431,459]
rows=[]
EDGES = [0, 1, 25, 75, 150, 300, 600, 1e9]
for d in days:
    x_cur = win_spend(d-27, d)          # 28 days ending at snapshot
    elig = first[first <= d-84].index   # households with a row at d
    # lagged pairs from snapshot d-28: x=spend[d-55,d-28], y=spend[d-27,d]
    x_prev = win_spend(d-55, d-28); y_prev = x_cur
    pool_prev = first[first <= d-112].index
    xp = x_prev.reindex(pool_prev).fillna(0.0).values
    yp = y_prev.reindex(pool_prev).fillna(0.0).values
    # fixed bins
    b = np.digitize(xp, EDGES)
    cal = {}
    for bi in np.unique(b):
        cal[bi] = yp[b==bi].mean()
    # second lagged pair d-56: x=spend[d-83,d-56], y=spend[d-55,d-28]
    x_prev2 = win_spend(d-83, d-56)
    pool2 = first[first <= d-140].index
    xp2 = x_prev2.reindex(pool2).fillna(0.0).values
    yp2 = x_prev.reindex(pool2).fillna(0.0).values
    b2 = np.digitize(xp2, EDGES); cal2={}
    for bi in np.unique(b2): cal2[bi]=yp2[b2==bi].mean()
    xc = x_cur.reindex(elig).fillna(0.0)
    bc = np.digitize(xc.values, EDGES)
    f1 = np.array([cal.get(bi, np.nan) for bi in bc])
    f2 = np.array([cal2.get(bi, np.nan) for bi in bc])
    # blended: weights 2:1 recent:older
    f12 = np.where(np.isnan(f2), f1, (2*f1+f2)/3.0)
    rows.append(pd.DataFrame({'household_key': elig, 'snapshot_day': d,
                              'cal1': f1, 'cal2': f2, 'cal_bl': f12, 'x_cur': xc.values}))
cal_df = pd.concat(rows, ignore_index=True)
print(cal_df.shape); print(cal_df.head())
print("bin means example d=431:", )
d=431; sub=cal_df[cal_df.snapshot_day==d]
for lo,hi in [(0,0),(1,25),(25,75),(75,150),(150,300),(300,600),(600,1e9)]:
    m=(sub.x_cur>=lo)&(sub.x_cur<hi) if hi<1e8 else sub.x_cur>=lo
    print(f"  bin[{lo},{hi}): n={m.sum():4d} cal1={sub.cal1[m].mean():7.1f} x_cur mean={sub.x_cur[m].mean():7.1f}")
agent_api.save_table(cal_df, "cal_v1_offline.parquet")


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
tt = agent_api.train_targets()
base = agent_api.load_saved("e013_stock.parquet")
cal  = agent_api.load_saved("cal_v1_offline.parquet")
keys=["household_key","snapshot_day"]
df = tt.merge(base,on=keys,how="left").merge(cal,on=keys,how="left")
cols0=[c for c in base.columns if c not in keys]
for c in cols0:
    if df[c].dtype=='object' or str(df[c].dtype)=='category':
        df[c]=df[c].astype('object').astype('category').cat.codes.astype(float).replace(-1,np.nan)

def ridge_eval(frame, feat_cols, tag, alphas=(8,16,32,64,128), verbose=False):
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
    if verbose:
        r=ph-yh; yh0=yh==0
        print("   zeros meanpred=%.1f MAE=%.1f | actives MAE=%.1f"%(ph[yh0].mean(),np.abs(r[yh0]).mean(),np.abs(r[~yh0]).mean()))
    print(f"[{tag}] alpha={a} inner={best[1]:.2f} hold431={mae:.3f}")
    return mae

m_ref=ridge_eval(df,cols0,"ref e013")
for add in [["cal1"],["cal1","cal2"],["cal_bl"],["cal1","cal2","cal_bl"]]:
    ridge_eval(df,cols0+add,"e013+"+",".join(add))
# calibration features ONLY combined with a few core levels (small model)
core=[c for c in cols0 if c in ['x_exp4w','spend_28','spend_84','spend_364','baskets_28','recency','x_spend_p1','x_spend_p2','active_28','x_bv_4']]
ridge_eval(df,core+["cal1","cal2","cal_bl"],"core+cal")
ridge_eval(df,["cal1","cal2","cal_bl"],"cal only",verbose=True)
ridge_eval(df,core,"core only")
# ratio to x_exp4w
df['cal_ratio']=df['cal_bl']/df['x_exp4w'].replace(0,np.nan)
df['cal_div']=df['cal_bl']-df['x_exp4w'].fillna(0)
ridge_eval(df,cols0+["cal1","cal2","cal_bl","cal_ratio","cal_div"],"e013+cal+ratios",verbose=True)


# ---- cell ----
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
print("x_exp4w describe:", df.x_exp4w.describe().round(2).to_dict())
print("recency describe:", df.recency.describe().round(1).to_dict())

def ridge_eval(frame, feat_cols, tag, alphas=(8,16,32,64,128), verbose=False):
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
    if verbose:
        r=ph-yh; yh0=yh==0
        print("   zeros meanpred=%.1f | actives MAE=%.1f"%(ph[yh0].mean(),np.abs(r[~yh0]).mean()))
    print(f"[{tag}] alpha={a} inner={best[1]:.2f} hold431={mae:.3f}")
    return mae

m_ref=ridge_eval(df,cols0,"ref e013")

def hinges(frame, col, edges, prefix):
    x = frame[col].fillna(0).clip(lower=0).values
    out = {}
    for e in edges:
        out[f"{prefix}h{int(e)}"] = np.maximum(0.0, x - e)
    return pd.DataFrame(out, index=frame.index)

# hinges on key level features (edges chosen from spend distribution)
edges_lvl = [10, 25, 50, 100, 200, 400, 800]
H = pd.concat([
    hinges(df,'x_exp4w',edges_lvl,'hx_'),
    hinges(df,'spend_84',edges_lvl,'hs84_'),
    hinges(df,'x_spend_p1',edges_lvl,'hp1_'),
    hinges(df,'x_spend_p2',edges_lvl,'hp2_'),
    hinges(df,'recency',[7,14,28,56,112],'hr_'),
], axis=1)
df2 = pd.concat([df, H], axis=1)
ridge_eval(df2, cols0+list(H.columns), "e013+hinges5", verbose=True)
# hinges only on x_exp4w
H1 = hinges(df,'x_exp4w',edges_lvl,'hx_')
df3 = pd.concat([df,H1],axis=1)
ridge_eval(df3, cols0+list(H1.columns), "e013+hinge_exp")
# fewer hinges
H2 = pd.concat([hinges(df,'x_exp4w',[25,100,300],'hx_'), hinges(df,'recency',[14,28],'hr_')],axis=1)
df4 = pd.concat([df,H2],axis=1)
ridge_eval(df4, cols0+list(H2.columns), "e013+hinges_few")


# ---- cell ----
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

def ridge_eval(frame, feat_cols, tag, alphas=(8,16,32,64,128), verbose=False):
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
    if verbose:
        r=ph-yh; yh0=yh==0
        print("   zeros meanpred=%.1f | actives MAE=%.1f"%(ph[yh0].mean(),np.abs(r[~yh0]).mean()))
    print(f"[{tag}] alpha={a} inner={best[1]:.2f} hold431={mae:.3f}")
    return mae

# ---- block-history shape features from raw transactions (offline, capped at 459) ----
v = agent_api.snapshot()
t = v.transactions[['household_key','day','sales_value']]
first = t.groupby('household_key').day.min().rename('first_day')
def win_spend(lo,hi):
    return t[(t.day>=lo)&(t.day<=hi)].groupby('household_key').sales_value.sum()
days=[95,123,151,179,207,235,263,291,319,347,375,403,431,459]
rows=[]
for d in days:
    elig=first[first<=d-84].index
    blocks={}
    for k in range(1,13):
        blocks[k]=win_spend(d-28*k+1, d-28*(k-1)).reindex(elig).fillna(0.0)
    B=pd.DataFrame(blocks)  # cols 1..12, block1 = most recent
    f=pd.DataFrame(index=elig)
    f['b_med6']=B[[1,2,3,4,5,6]].median(axis=1)
    f['b_med12']=B.median(axis=1)
    f['b_mean6']=B[[1,2,3,4,5,6]].mean(axis=1)
    f['b_mean12']=B.mean(axis=1)
    f['b_pos_share12']=(B>0).mean(axis=1)
    f['b_pos_share6']=(B[[1,2,3,4,5,6]]>0).mean(axis=1)
    zs=0
    for k in range(1,13):
        zs=zs+(B[k]==0)*1*0  # placeholder
    # consecutive zero blocks ending at block1
    zst=np.zeros(len(elig))
    Bv=B.values
    for i in range(len(elig)):
        c=0
        for k in range(12):
            if Bv[i,k]==0: c+=1
            else: break
        zst[i]=c
    f['b_zerostreak']=zst
    f['b_max12']=B.max(axis=1)
    f['b_std12']=B.std(axis=1)
    f['b_med6_x_pos']=f['b_med6']*f['b_pos_share6']
    f['b_mean_pos']=B.replace(0,np.nan).mean(axis=1).fillna(0)
    f['b_med_div_mean']=f['b_med6']/f['b_mean6'].replace(0,np.nan)
    rows.append(f.reset_index().rename(columns={'index':'household_key'}).assign(snapshot_day=d))
shape=pd.concat(rows,ignore_index=True)
agent_api.save_table(shape,"shape_v1_offline.parquet")
print("shape",shape.shape)

df2 = tt.merge(base,on=keys,how="left").merge(shape,on=keys,how="left")
scols=[c for c in shape.columns if c not in keys+['index']]
m_ref=ridge_eval(df2,cols0,"ref")
ridge_eval(df2,cols0+scols,"e013+shape",verbose=True)
ridge_eval(df2,scols,"shape only")
# small: core + shape
core=[c for c in cols0 if c in ['x_exp4w','spend_28','spend_84','spend_364','baskets_28','recency','x_spend_p1','x_spend_p2','active_28','x_bv_4','x_r_4_8']]
ridge_eval(df2,core+scols,"core+shape")
ridge_eval(df2,core,"core")


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
tt = agent_api.train_targets()
base = agent_api.load_saved("e013_stock.parquet")
shape = agent_api.load_saved("shape_v1_offline.parquet")
keys=["household_key","snapshot_day"]
df2 = tt.merge(base,on=keys,how="left").merge(shape,on=keys,how="left")
cols0=[c for c in base.columns if c not in keys]
for c in cols0:
    if df2[c].dtype=='object' or str(df2[c].dtype)=='category':
        df2[c]=df2[c].astype('object').astype('category').cat.codes.astype(float).replace(-1,np.nan)
scols=[c for c in shape.columns if c not in keys+['index']]

def ridge_eval(frame, feat_cols, tag, alphas=(8,16,32,64,128), verbose=False):
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
    if verbose:
        r=ph-yh; yh0=yh==0
        print("   zeros meanpred=%.1f | actives MAE=%.1f"%(ph[yh0].mean(),np.abs(r[~yh0]).mean()))
    print(f"[{tag}] alpha={a} inner={best[1]:.2f} hold431={mae:.3f}")
    return mae

m_ref=ridge_eval(df2,cols0,"ref")
ridge_eval(df2,cols0+scols,"e013+shape",verbose=True)
ridge_eval(df2,scols,"shape only")
core=[c for c in cols0 if c in ['x_exp4w','spend_28','spend_84','spend_364','baskets_28','recency','x_spend_p1','x_spend_p2','active_28','x_bv_4','x_r_4_8']]
ridge_eval(df2,core+scols,"core+shape")
ridge_eval(df2,core,"core")

# ---- GBM ceiling check: tiny histogram GBM on e013 features ----
def gbm_eval(frame, feat_cols, tag, n_rounds=400, lr=0.05, depth=6, seed=0):
    fit=frame[frame.snapshot_day<=403]; hold=frame[frame.snapshot_day==431]
    yf=fit.future_spend_4w.values; yh=hold.future_spend_4w.values
    def prep(X):
        X=X.astype(float).copy()
        med=np.nanmedian(X,axis=0)
        return np.where(np.isnan(X),med,X)
    Xf=prep(fit[feat_cols].values); Xh=prep(hold[feat_cols].values)
    # log-target GBM
    lf=np.log1p(yf)
    pred_f=np.zeros(len(fit)); pred_h=np.zeros(len(hold))
    feat_idx=np.arange(Xf.shape[1])
    rng=np.random.RandomState(seed)
    nfeat=min(40,len(feat_idx))
    for it in range(n_rounds):
        r=lf-pred_f
        feats=rng.choice(feat_idx,nfeat,replace=False)
        best=None
        for j in feats:
            order=np.argsort(Xf[:,j])
            xs=Xf[order,j]; rs=r[order]
            # candidate thresholds at quantiles
            qs=np.unique(np.quantile(xs,[0.1,0.25,0.5,0.75,0.9]))
            for q in qs:
                m=xs<=q
                if m.sum()<50 or (~m).sum()<50: continue
                gl=rs[m].sum(); gr=rs[~m].sum()
                gain=gl*gl/(m.sum()+1e-9)+gr*gr/((~m).sum()+1e-9)
                if best is None or gain>best[0]: best=(gain,j,q,m)
        if best is None: break
        _,j,q,m=best
        lv=r[m].mean(); rv=r[~m].mean()
        pred_f+=lr*np.where(m,lv,rv)
        pred_h+=lr*np.where(Xh[:,j]<=q,lv,rv)
    ph=np.clip(np.expm1(pred_h),0,None)
    mae=np.abs(ph-yh).mean()
    print(f"[GBM {tag}] hold431={mae:.3f}  zeros meanpred={ph[yh==0].mean():.1f}")
    return mae
gbm_eval(df2,cols0,"e013 log-target")
gbm_eval(df2,cols0+scols,"e013+shape log-target")
gbm_eval(df2,core+scols,"core+shape log-target")


# ---- cell ----
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

def ridge_eval(frame, feat_cols, tag, alphas=(8,16,32,64,128,256), verbose=False):
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
    if verbose:
        r=ph-yh; yh0=yh==0
        print("   zeros meanpred=%.1f | actives MAE=%.1f"%(ph[yh0].mean(),np.abs(r[~yh0]).mean()))
    print(f"[{tag}] alpha={a} inner={best[1]:.2f} hold431={mae:.3f}")
    return mae

def hinges(frame, col, edges, prefix, fill=0.0):
    x = frame[col].fillna(fill).clip(lower=0).values.astype(float)
    return pd.DataFrame({f"{prefix}h{int(e)}": np.maximum(0.0, x-e) for e in edges}, index=frame.index)

H = pd.concat([
    hinges(df,'x_exp4w',[10,25,50,100,200,400,800],'hx_'),
    hinges(df,'spend_84',[25,100,300],'hs84_'),
    hinges(df,'x_spend_p1',[10,25,50,100,200,400,800],'hp1_'),
    hinges(df,'x_spend_p2',[10,25,50,100,200,400,800],'hp2_'),
    hinges(df,'recency',[7,14,28,56,112],'hr_'),
    hinges(df,'x_spend_12',[25,100,300],'hs12_'),
    hinges(df,'baskets_28',[1,3,6,10],'hb28_'),
    hinges(df,'x_bv_4',[10,25,50,100],'hbv_'),
    hinges(df,'days_28',[1,4,8,14],'hd28_'),
],axis=1)
dfH = pd.concat([df,H],axis=1)
ridge_eval(dfH,cols0,"ref")
ridge_eval(dfH,cols0+list(H.columns),"e013+hinges_ext",verbose=True)
# pruning: drop weak blocks
dropmix=[c for c in cols0 if c.startswith('p_') or c.startswith('coh_') or c.startswith('stk_') or c in ('dow_entropy','modal_dow','zero_w12','wk_cv','unit_price','n_prod84','p_other')]
keep=[c for c in cols0 if c not in dropmix]
ridge_eval(dfH,keep+list(H.columns),"pruned+hinges_ext",verbose=True)
ridge_eval(dfH,keep,"pruned")


# ---- cell ----
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

def ridge_eval(frame, feat_cols, tag, alphas=(32,64,128,256,512), verbose=False):
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
    if verbose:
        r=ph-yh; yh0=yh==0
        print("   zeros meanpred=%.1f | actives MAE=%.1f"%(ph[yh0].mean(),np.abs(r[~yh0]).mean()))
    print(f"[{tag}] alpha={a} inner={best[1]:.2f} hold431={mae:.3f}")
    return mae

m_ref=ridge_eval(df,cols0,"ref")

# ---- one-hot bins of key levels ----
def onehot_bins(frame, col, edges, prefix):
    x=frame[col].fillna(0).clip(lower=0).values.astype(float)
    b=np.digitize(x,edges)
    D=pd.get_dummies(b, prefix=prefix).astype(float)
    D.index=frame.index
    return D
edges_lvl=[0,10,25,50,75,100,150,200,300,450,600]
OH = pd.concat([
    onehot_bins(df,'x_exp4w',edges_lvl,'ohx'),
    onehot_bins(df,'spend_84',edges_lvl,'ohs84'),
    onehot_bins(df,'recency',[1,7,14,21,28,56,112],'ohr'),
    onehot_bins(df,'baskets_28',[0,1,2,4,7,11],'ohb'),
],axis=1)
dfO=pd.concat([df,OH],axis=1)
ridge_eval(dfO,cols0+list(OH.columns),"e013+onehot_bins",verbose=True)
# onehot of x_exp4w only
OHx=onehot_bins(df,'x_exp4w',edges_lvl,'ohx')
dfO2=pd.concat([df,OHx],axis=1)
ridge_eval(dfO2,cols0+list(OHx.columns),"e013+oh_exp")
# recency onehot only
OHr=onehot_bins(df,'recency',[1,7,14,21,28,56,112],'ohr')
dfO3=pd.concat([df,OHr],axis=1)
ridge_eval(dfO3,cols0+list(OHr.columns),"e013+oh_rec")

# ---- department-level spend features (offline from capped view; valid for hold431/inner403) ----
v=agent_api.snapshot()
t=v.transactions[['household_key','day','sales_value','product_id']].merge(
    v.products[['product_id','department']],on='product_id',how='left')
CONSUM={'GROCERY','MEAT','PRODUCE','DELI','MEAT-PCKGD','SEAFOOD','SEAFOOD-PCKGD','PASTRY','DAIRY','FROZEN','BAKERY','KIOSK-GAS'}
t['is_cons']=t.department.isin(CONSUM).astype(float)
t['dept8']=t.department.where(t.department.isin(['GROCERY','DRUG GM','MEAT','PRODUCE','MEAT-PCKGD','DELI','KIOSK-GAS','PASTRY']),'other')
days=[95,123,151,179,207,235,263,291,319,347,375,403,431,459]
first=t.groupby('household_key').day.min()
rows=[]
for d in days:
    elig=first[first<=d-84].index
    w28=t[(t.day>=d-27)&(t.day<=d)]; w84=t[(t.day>=d-83)&(t.day<=d)]
    f=pd.DataFrame(index=elig)
    s28=w28.groupby(['household_key','dept8']).sales_value.sum().unstack(fill_value=0.0)
    s84=w84.groupby(['household_key','dept8']).sales_value.sum().unstack(fill_value=0.0)
    s28=s28.reindex(elig).fillna(0.0); s84=s84.reindex(elig).fillna(0.0)
    for c in s28.columns:
        f[f'd28_{c}']=s28[c].values
        f[f'r2884_{c}']=(s28[c]/s84[c].replace(0,np.nan)).fillna(1.0).values
    c28=w28.groupby('household_key').is_cons.sum().reindex(elig).fillna(0.0)
    a28=w28.groupby('household_key').sales_value.sum().reindex(elig).fillna(0.0)
    f['cons_share28']=(c28/a28.replace(0,np.nan)).fillna(0).values
    c84=w84.groupby('household_key').is_cons.sum().reindex(elig).fillna(0.0)
    a84=w84.groupby('household_key').sales_value.sum().reindex(elig).fillna(0.0)
    f['cons_share84']=(c84/a84.replace(0,np.nan)).fillna(0).values
    # days since last consumable purchase
    lastc=w28[w28.is_cons==1].groupby('household_key').day.max()
    f['days_since_cons']=(d-lastc).reindex(elig).fillna(28).values
    rows.append(f.reset_index().rename(columns={'index':'household_key'}).assign(snapshot_day=d))
dept=pd.concat(rows,ignore_index=True)
agent_api.save_table(dept,"dept_v1_offline.parquet")
print("dept feats:",dept.shape)
dfD=tt.merge(base,on=keys,how="left").merge(dept,on=keys,how="left")
dcols=[c for c in dept.columns if c not in keys+['index']]
ridge_eval(dfD,cols0+dcols,"e013+dept",verbose=True)
ridge_eval(dfD,cols0+dcols+list(OH.columns),"e013+dept+oh",verbose=True)


# ---- cell ----
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

def ridge_eval(frame, feat_cols, tag, alphas=(32,64,128,256,512), verbose=False):
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
    if verbose:
        r=ph-yh; yh0=yh==0
        print("   zeros meanpred=%.1f | actives MAE=%.1f"%(ph[yh0].mean(),np.abs(r[~yh0]).mean()))
    print(f"[{tag}] alpha={a} inner={best[1]:.2f} hold431={mae:.3f}")
    return mae

v=agent_api.snapshot()
t=v.transactions[['household_key','day','sales_value','product_id']].merge(
    v.products[['product_id','department']].astype({'department':'str'}),on='product_id',how='left')
t['department']=t['department'].fillna('other').astype(str)
CONSUM={'GROCERY','MEAT','PRODUCE','DELI','MEAT-PCKGD','SEAFOOD','SEAFOOD-PCKGD','PASTRY','DAIRY','FROZEN','BAKERY','KIOSK-GAS'}
t['is_cons']=t.department.isin(CONSUM).astype(float)
t['dept8']=np.where(t.department.isin(['GROCERY','DRUG GM','MEAT','PRODUCE','MEAT-PCKGD','DELI','KIOSK-GAS','PASTRY']),t.department,'other')
days=[95,123,151,179,207,235,263,291,319,347,375,403,431,459]
first=t.groupby('household_key').day.min()
rows=[]
for d in days:
    elig=first[first<=d-84].index
    w28=t[(t.day>=d-27)&(t.day<=d)]; w84=t[(t.day>=d-83)&(t.day<=d)]
    f=pd.DataFrame(index=elig)
    s28=w28.groupby(['household_key','dept8']).sales_value.sum().unstack(fill_value=0.0)
    s84=w84.groupby(['household_key','dept8']).sales_value.sum().unstack(fill_value=0.0)
    s28=s28.reindex(elig).fillna(0.0); s84=s84.reindex(elig).fillna(0.0)
    for c in s28.columns:
        f[f'd28_{c}']=s28[c].values
        f[f'r2884_{c}']=(s28[c]/s84[c].replace(0,np.nan)).fillna(1.0).values
    c28=w28.groupby('household_key').is_cons.sum().reindex(elig).fillna(0.0)
    a28=w28.groupby('household_key').sales_value.sum().reindex(elig).fillna(0.0)
    f['cons_share28']=(c28/a28.replace(0,np.nan)).fillna(0).values
    c84=w84.groupby('household_key').is_cons.sum().reindex(elig).fillna(0.0)
    a84=w84.groupby('household_key').sales_value.sum().reindex(elig).fillna(0.0)
    f['cons_share84']=(c84/a84.replace(0,np.nan)).fillna(0).values
    lastc=w28[w28.is_cons==1].groupby('household_key').day.max()
    f['days_since_cons']=(d-lastc).reindex(elig).fillna(28).values
    rows.append(f.reset_index().rename(columns={'index':'household_key'}).assign(snapshot_day=d))
dept=pd.concat(rows,ignore_index=True)
agent_api.save_table(dept,"dept_v1_offline.parquet")
print("dept feats:",dept.shape, list(dept.columns)[:8])
dfD=tt.merge(base,on=keys,how="left").merge(dept,on=keys,how="left")
dcols=[c for c in dept.columns if c not in keys+['index']]
ridge_eval(dfD,cols0+dcols,"e013+dept",verbose=True)


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
tt = agent_api.train_targets()
base = agent_api.load_saved("e013_stock.parquet")
dept = agent_api.load_saved("dept_v1_offline.parquet")
keys=["household_key","snapshot_day"]

def enc(frame, cols):
    for c in cols:
        if frame[c].dtype=='object' or str(frame[c].dtype)=='category':
            frame[c]=frame[c].astype('object').astype('category').cat.codes.astype(float).replace(-1,np.nan)
    return frame

def ridge_eval(frame, feat_cols, tag, alphas=(32,64,128,256,512), verbose=False):
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
    if verbose:
        r=ph-yh; yh0=yh==0
        print("   zeros meanpred=%.1f | actives MAE=%.1f"%(ph[yh0].mean(),np.abs(r[~yh0]).mean()))
    print(f"[{tag}] alpha={a} inner={best[1]:.2f} hold431={mae:.3f}")
    return mae

cols0=[c for c in base.columns if c not in keys]
dfD = enc(tt.merge(base,on=keys,how="left").merge(dept,on=keys,how="left"), cols0+[c for c in dept.columns if c not in keys+['index']])
dcols=[c for c in dept.columns if c not in keys+['index']]
ridge_eval(dfD,cols0,"ref")
ridge_eval(dfD,cols0+dcols,"e013+dept",verbose=True)

# ---------------- two-part model from lagged pairs ----------------
v=agent_api.snapshot()
t=v.transactions[['household_key','day','sales_value']]
first=t.groupby('household_key').day.min().rename('first_day')
def spend_win(lo,hi,pool=None):
    s=t[(t.day>=lo)&(t.day<=hi)].groupby('household_key').sales_value.sum()
    if pool is not None: s=s.reindex(pool)
    return s.fillna(0.0)
def hist_feats(a, pool):
    tp=t[t.day<=a]
    last=tp.groupby('household_key').day.max()
    rec=(a-last).reindex(pool).astype(float)
    tw=tp[(tp.day>=a-83)].copy()
    tw['w']=np.exp(-(a-tw.day)/40.0)
    xexp=tw.assign(sv=tw.sales_value*tw.w).groupby('household_key').sv.sum().reindex(pool).fillna(0.0)
    pos=np.zeros(len(pool)); zs=np.zeros(len(pool))
    bl=[]
    for k in range(1,13):
        bl.append(spend_win(a-28*k+1,a-28*(k-1),pool).values)
    B=np.array(bl).T
    pos=(B>0).mean(axis=1)
    zs=np.zeros(len(pool))
    for i in range(len(pool)):
        c=0
        for k in range(12):
            if B[i,k]==0: c+=1
            else: break
        zs[i]=c
    return pd.DataFrame({'rec':rec.values,'xexp':xexp.values,'pos':pos,'zst':zs},index=pool)

EDGES=[0,10,25,50,75,100,150,200,300,450,600]
def fit_logistic(X,y,lam=1.0,iters=25):
    mu=X.mean(0); sd=X.std(0)+1e-9; Z=(X-mu)/sd; Z=np.hstack([Z,np.ones((len(Z),1))])
    w=np.zeros(Z.shape[1])
    for _ in range(iters):
        p=1/(1+np.exp(-Z@w))
        g=Z.T@(p-y)-lam*w; H=Z.T@(Z*(p*(1-p))[:,None])+lam*np.eye(Z.shape[1])
        step=np.linalg.solve(H,g); w=w-step
        if np.abs(step).max()<1e-6: break
    return w,mu,sd
rows=[]
days=[95,123,151,179,207,235,263,291,319,347,375,403,431,459]
for d in days:
    elig=first[first<=d-84].index
    parts=[]
    for lag,ylo,yhi in [(28,d-27,d),(56,d-55,d-28)]:
        a=d-lag
        pool=first[first<=a-84].index
        if len(pool)<200:
            pool=first[first<=a-28].index
        if len(pool)<200: parts.append(None); continue
        Hf=hist_feats(a,pool)
        yact=(spend_win(ylo,yhi,pool).values>0).astype(float)
        yspend=spend_win(ylo,yhi,pool).values
        w,mu,sd=fit_logistic(Hf[['rec','xexp','pos']].values,yact)
        Hc=hist_feats(d,elig)
        Z=(Hc[['rec','xexp','pos']].values-mu)/sd; Z=np.hstack([Z,np.ones((len(Z),1))])
        p=np.clip(1/(1+np.exp(-Z@w)),0.02,0.98)
        act=yspend>0
        b=np.digitize(Hc['xexp'].values,EDGES)
        cal=np.full(len(elig),np.nan)
        for bi in np.unique(b):
            m=b==bi
            sel=(np.digitize(Hf['xexp'].values,EDGES)==bi)&act
            if sel.sum()>=30: cal[m]=yspend[sel].mean()
        parts.append(pd.DataFrame({'p_act':p,'cal':cal},index=elig))
    f=pd.DataFrame(index=elig)
    p1=parts[0]['p_act']; c1=parts[0]['cal']
    p2=parts[1]['p_act'] if parts[1] is not None else p1
    c2=parts[1]['cal'] if parts[1] is not None else c1
    f['tp_p']=((p1+p2)/2).values
    calbl=pd.concat([c1,c2],axis=1).mean(axis=1)
    f['tp_f']= (f['tp_p']*calbl.fillna(0)).values
    f['tp_px']= (f['tp_p']*base.set_index(keys).reindex(pd.MultiIndex.from_arrays([elig,[d]*len(elig)]))['x_exp4w'].fillna(0)).values if False else (f['tp_p'].values*0)
    rows.append(f.reset_index().rename(columns={'index':'household_key'}).assign(snapshot_day=d))
tp=pd.concat(rows,ignore_index=True)
print("tp",tp.shape, tp.tp_p.describe().round(3).to_dict())
agent_api.save_table(tp,"tp_v1_offline.parquet")
dfT=enc(tt.merge(base,on=keys,how="left").merge(tp,on=keys,how="left"),cols0)
tcols=[c for c in tp.columns if c not in keys+['index']]
ridge_eval(dfT,cols0+tcols,"e013+twopart",verbose=True)
ridge_eval(dfT,cols0+["tp_f"],"e013+tp_f")
ridge_eval(dfT,cols0+["tp_p"],"e013+tp_p")


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
tt = agent_api.train_targets()
base = agent_api.load_saved("e013_stock.parquet")
dept = agent_api.load_saved("dept_v1_offline.parquet")
struct = agent_api.load_saved("structure_v1.parquet")
keys=["household_key","snapshot_day"]

def enc(frame, cols):
    for c in cols:
        if c in frame.columns and (frame[c].dtype=='object' or str(frame[c].dtype)=='category'):
            frame[c]=frame[c].astype('object').astype('category').cat.codes.astype(float).replace(-1,np.nan)
    return frame

def ridge_eval(frame, feat_cols, tag, alphas=(64,128,256,512), verbose=False):
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
    if verbose:
        r=ph-yh; yh0=yh==0
        print("   zeros meanpred=%.1f | actives MAE=%.1f"%(ph[yh0].mean(),np.abs(r[~yh0]).mean()))
    print(f"[{tag}] alpha={a} inner={best[1]:.2f} hold431={mae:.3f}")
    return mae

cols0=[c for c in base.columns if c not in keys]
dfm = enc(tt.merge(base,on=keys,how="left").merge(dept,on=keys,how="left").merge(struct,on=keys,how="left"),
          cols0+[c for c in dept.columns if c not in keys+['index']]+[c for c in struct.columns if c not in keys+['index']])
dcols=[c for c in dept.columns if c not in keys+['index']]
scols=[c for c in struct.columns if c not in keys+['index']]

def hinges(frame, col, edges, prefix, fill=0.0):
    x = frame[col].fillna(fill).clip(lower=0).values.astype(float)
    return pd.DataFrame({f"{prefix}h{int(e)}": np.maximum(0.0, x-e) for e in edges}, index=frame.index)
H = pd.concat([
    hinges(dfm,'x_exp4w',[10,25,50,100,200,400,800],'hx_'),
    hinges(dfm,'spend_84',[25,100,300],'hs84_'),
    hinges(dfm,'x_spend_p1',[10,25,50,100,200,400,800],'hp1_'),
    hinges(dfm,'x_spend_p2',[10,25,50,100,200,400,800],'hp2_'),
    hinges(dfm,'recency',[7,14,28,56,112],'hr_'),
    hinges(dfm,'x_spend_12',[25,100,300],'hs12_'),
    hinges(dfm,'baskets_28',[1,3,6,10],'hb28_'),
    hinges(dfm,'x_bv_4',[10,25,50,100],'hbv_'),
    hinges(dfm,'days_28',[1,4,8,14],'hd28_'),
],axis=1)
def onehot_bins(frame, col, edges, prefix):
    x=frame[col].fillna(0).clip(lower=0).values.astype(float)
    b=np.digitize(x,edges)
    D=pd.get_dummies(b, prefix=prefix).astype(float); D.index=frame.index
    return D
OH = onehot_bins(dfm,'x_exp4w',[0,10,25,50,75,100,150,200,300,450,600],'ohx')
dfm2=pd.concat([dfm,H,OH],axis=1)
Hc=list(H.columns); OHc=list(OH.columns)
ridge_eval(dfm2,cols0,"ref")
ridge_eval(dfm2,cols0+Hc,"A h")
ridge_eval(dfm2,cols0+Hc+OHc,"B h+oh")
ridge_eval(dfm2,cols0+Hc+OHc+dcols,"C h+oh+dept")
ridge_eval(dfm2,cols0+Hc+OHc+dcols+scols,"D h+oh+dept+struct",verbose=True)
ridge_eval(dfm2,cols0+scols,"E struct only")
ridge_eval(dfm2,cols0+dcols+scols,"F dept+struct")


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

DEPTS=['GROCERY','DRUG GM','MEAT','PRODUCE','MEAT-PCKGD','DELI','KIOSK-GAS','PASTRY']

def fn(view, snapshot_day):
    d = snapshot_day
    hh = pd.Index(view.households)
    try:
        b = agent_api.load_saved("e013_stock.parquet")
        sub = b[b.snapshot_day==d].set_index('household_key')
        X = sub.drop(columns=['snapshot_day']).reindex(hh)
        def hg(col, edges, pfx, fill=0.0):
            x = X[col].fillna(fill).clip(lower=0).astype(float).values
            for e in edges: X[f"{pfx}h{int(e)}"] = np.maximum(0.0, x-e)
        hg('x_exp4w',[10,25,50,100,200,400,800],'hx_')
        hg('spend_84',[25,100,300],'hs84_')
        hg('x_spend_p1',[10,25,50,100,200,400,800],'hp1_')
        hg('x_spend_p2',[10,25,50,100,200,400,800],'hp2_')
        hg('recency',[7,14,28,56,112],'hr_')
        hg('x_spend_12',[25,100,300],'hs12_')
        hg('baskets_28',[1,3,6,10],'hb28_')
        hg('x_bv_4',[10,25,50,100],'hbv_')
        hg('days_28',[1,4,8,14],'hd28_')
        x = X['x_exp4w'].fillna(0).clip(lower=0).values.astype(float)
        bns = np.digitize(x, [0,10,25,50,75,100,150,200,300,450,600])
        for k in range(12): X[f'ohx_{k}'] = (bns==k).astype(float)
        t = view.transactions[['household_key','day','sales_value','product_id']].merge(
            view.products[['product_id','department']].astype({'department':'str'}), on='product_id', how='left')
        t['department']=t['department'].fillna('other').astype(str)
        t['dept8']=np.where(t.department.isin(DEPTS), t.department, 'other')
        w28=t[(t.day>=d-27)&(t.day<=d)]; w84=t[(t.day>=d-83)&(t.day<=d)]
        s28=w28.groupby(['household_key','dept8']).sales_value.sum().unstack(fill_value=0.0).reindex(hh).fillna(0.0)
        s84=w84.groupby(['household_key','dept8']).sales_value.sum().unstack(fill_value=0.0).reindex(hh).fillna(0.0)
        for c in DEPTS+['other']:
            X[f'd28_{c}'] = s28[c].values if c in s28.columns else 0.0
            if c in s28.columns and c in s84.columns:
                X[f'r2884_{c}'] = (s28[c]/s84[c].replace(0,np.nan)).fillna(1.0).values
            else:
                X[f'r2884_{c}'] = 1.0
        CONSUM={'GROCERY','MEAT','PRODUCE','DELI','MEAT-PCKGD','SEAFOOD','SEAFOOD-PCKGD','PASTRY','DAIRY','FROZEN','BAKERY','KIOSK-GAS'}
        t['is_cons']=t.department.isin(CONSUM).astype(float)
        c28=w28.groupby('household_key').is_cons.sum().reindex(hh).fillna(0.0)
        a28=w28.groupby('household_key').sales_value.sum().reindex(hh).fillna(0.0)
        X['cons_share28']=(c28/a28.replace(0,np.nan)).fillna(0).values
        c84=w84.groupby('household_key').is_cons.sum().reindex(hh).fillna(0.0)
        a84=w84.groupby('household_key').sales_value.sum().reindex(hh).fillna(0.0)
        X['cons_share84']=(c84/a84.replace(0,np.nan)).fillna(0).values
        lastc=w28[w28.is_cons==1].groupby('household_key').day.max()
        X['days_since_cons']=(d-lastc).reindex(hh).fillna(28).astype(float).values
        return X
    except Exception as e:
        print("FN-ERR", d, repr(e)[:200])
        F=pd.DataFrame(index=hh)
        tr=view.transactions
        F['spend_28']=tr[(tr.day>=d-27)].groupby('household_key').sales_value.sum().reindex(hh).fillna(0.0)
        F['spend_84']=tr[(tr.day>=d-83)].groupby('household_key').sales_value.sum().reindex(hh).fillna(0.0)
        F['baskets_28']=tr[(tr.day>=d-27)].groupby('household_key').basket_id.nunique().reindex(hh).fillna(0.0)
        F['recency']=(d-tr.groupby('household_key').day.max()).reindex(hh).astype(float)
        F['tenure']=(d-tr.groupby('household_key').day.min()).reindex(hh).astype(float)
        return F

out = agent_api.build_features(fn)
print("built:", out.shape)
newcols=[c for c in out.columns if c.startswith(('hx_','hs84_','hp1_','hp2_','hr_','hs12_','hb28_','hbv_','hd28_','ohx_','d28_','r2884_','cons_','days_since'))]
print("n new cols:", len(newcols), "| sample:", newcols[:6], newcols[-4:])
print("NaN frac in new cols: %.4f" % np.mean([out[c].isna().mean() for c in newcols]))
print("snapshots:", sorted(out.snapshot_day.unique()))
p = agent_api.save_table(out, "e016_flex.parquet")
print("saved:", p)
