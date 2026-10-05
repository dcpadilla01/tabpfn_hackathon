import agent_api as A, pandas as pd, numpy as np
for name in ["e005_preds","e005_newfeats","e004_features","e004_new","e002_features","e004_preds","e003_preds"]:
    try:
        df = A.load_saved(name+".parquet")
        print(name, df.shape, list(df.columns)[:12], "..." if df.shape[1]>12 else "")
    except Exception as e:
        print(name, "ERR", type(e).__name__, e)
tt = A.train_targets()
print("targets", tt.shape, tt.columns.tolist())
print("zero frac", (tt.future_spend_4w==0).mean(), "mean", tt.future_spend_4w.mean(), "median", tt.future_spend_4w.median(), "p90", tt.future_spend_4w.quantile(.9))
print(A.snapshot_days())


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
tt = A.train_targets()
e5 = A.load_saved("e005_preds.parquet")
# per-snapshot-day target stats (train)
g = tt.groupby("snapshot_day").future_spend_4w.agg(["mean","median",lambda s:(s==0).mean(),"count"])
g.columns=["mean","median","zerofrac","n"]; print("TRAIN targets by day\n", g.round(1))
# e5 predictions by snapshot day (validation only in preds? check)
print("pred days:", e5.snapshot_day.unique())
pv = e5.groupby("snapshot_day").prediction.agg(["mean","median",lambda s:(s<=0).mean(),"count"])
pv.columns=["mean","median","zerofrac","n"]; print("E5 preds by day\n", pv.round(1))
# blend check: correlation of e3/e4/e5
m = e5.merge(A.load_saved("e004_preds.parquet"), on=["household_key","snapshot_day"], suffixes=("_e5","_e4")).merge(A.load_saved("e003_preds.parquet").rename(columns={"prediction":"pred_e3"}), on=["household_key","snapshot_day"])
print(m[["prediction_e5","prediction_e4","pred_e3"]].corr().round(4))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
f4 = A.load_saved("e004_features.parquet")
f5 = A.load_saved("e005_newfeats.parquet")
tt = A.train_targets()
print("e004 cols:\n", [c for c in f4.columns])
print("e005 cols:\n", [c for c in f5.columns])
m = f4.merge(f5.drop(columns=[c for c in f5.columns if c in f4.columns or c in ("index",)]), on=["household_key","snapshot_day"], how="inner") if "index" in f4.columns else None
print("shapes", f4.shape, f5.shape)


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time, xgboost as xgb
f4 = A.load_saved("e004_features.parquet"); f5 = A.load_saved("e005_newfeats.parquet")
F = f4.merge(f5.drop(columns=["household_key","snapshot_day"]), left_index=True, right_index=True) if False else f4.merge(f5, on=["household_key","snapshot_day"], how="inner")
print("F", F.shape)
tt = A.train_targets()
Ft = F.merge(tt, on=["household_key","snapshot_day"])
print("Ft", Ft.shape)
feats = [c for c in F.columns if c not in ("household_key","snapshot_day")]
# extra cheap ratio features
Ft["r28_364"] = Ft.spend_28/(Ft.spend_364/13+1e-6)
Ft["r7_364"]  = Ft.spend_7/(Ft.spend_364/52+1e-6)
Ft["r84_364"] = Ft.spend_84/(Ft.spend_364/4+1e-6)
Ft["act_ratio_28_84"] = Ft.act_28/(Ft.act_84+1e-6)
feats += ["r28_364","r7_364","r84_364","act_ratio_28_84"]
feats = [c for c in feats if c in Ft.columns]
Xall = Ft[feats].astype("float32"); y = Ft.future_spend_4w.values.astype("float32"); day = Ft.snapshot_day.values

def run_xgb(Xtr,ytr,wtr,Xte,params,rounds,seeds=(11,22,33)):
    dtr = xgb.DMatrix(Xtr,label=ytr,weight=wtr); dte = xgb.DMatrix(Xte)
    ps = dict(params); ps["objective"]="reg:quantileerror"; ps["quantile_alpha"]=0.5
    preds=[]
    for s in seeds:
        ps["seed"]=s
        b = xgb.train(ps,dtr,rounds)
        preds.append(b.predict(dte))
    return np.mean(preds,axis=0)

base = dict(learning_rate=0.03,max_depth=6,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,reg_lambda=1.0,tree_method="hist",eval_metric="quantile")
t0=time.time()
for hold in [347,403,431]:
    tr = day<hold; te = day==hold
    w = 0.5**((hold-day[tr])/140.0)
    p = run_xgb(Xall[tr],y[tr],w,Xall[te],base,1200)
    mae = np.abs(p-y[te]).mean()
    print(f"hold {hold}: n_tr={tr.sum()} MAE={mae:.3f}  ({time.time()-t0:.0f}s)")


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
tt = A.train_targets(); F = A.load_saved("e004_features.parquet")
d = F.merge(tt, on=["household_key","snapshot_day"])
# naive predictors
for c in ["spend_28","spend_84","spend_112"]:
    d["p_"+c] = d[c]* (28/ (28 if c=="spend_28" else 84 if c=="spend_84" else 112))
    print(c, "MAE", np.abs(d.p_ - d.future_spend_4w if False else d["p_"+c]-d.future_spend_4w).mean().round(2))
# blend naive with 0.5 weight
d["p_mix"] = 0.5*d.p_spend_28 + 0.5*d.p_spend_84
print("mix MAE", np.abs(d.p_mix-d.future_spend_4w).mean().round(2))
# residual structure of e5 on train? we don't have train preds saved. Use holdout-style: check target vs spend_28 quantiles
d["b"] = pd.qcut(d.spend_28, 10, duplicates="drop")
print(d.groupby("b", observed=True).agg(mean_t=("future_spend_4w","mean"), mean_s28=("spend_28","mean"), n=("future_spend_4w","size")).round(1))
# zero-households: what do they buy after a 0-spend 28d window?
z = d[d.spend_28<=1]
print("spend28<=1: n", len(z), "mean target", z.future_spend_4w.mean().round(1), "zerofrac", (z.future_spend_4w==0).mean().round(2))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time, xgboost as xgb
f4 = A.load_saved("e004_features.parquet"); f5 = A.load_saved("e005_newfeats.parquet")
F = f4.merge(f5, on=["household_key","snapshot_day"], how="inner")
tt = A.train_targets(); Ft = F.merge(tt, on=["household_key","snapshot_day"])
feats = [c for c in F.columns if c not in ("household_key","snapshot_day")]
Ft["r28_364"]=Ft.spend_28/(Ft.spend_364/13+1e-6); Ft["r7_364"]=Ft.spend_7/(Ft.spend_364/52+1e-6)
Ft["r84_364"]=Ft.spend_84/(Ft.spend_364/4+1e-6); Ft["act_ratio_28_84"]=Ft.act_28/(Ft.act_84+1e-6)
feats += ["r28_364","r7_364","r84_364","act_ratio_28_84"]
X = Ft[feats].astype("float32"); y = Ft.future_spend_4w.values.astype("float32"); day = Ft.snapshot_day.values
# global mean target by day: 130->146 rising. Try adding snapshot_day itself as feature
Ft["sd"] = Ft.snapshot_day
feats2 = feats+["sd"]

def run(Xtr,ytr,wtr,Xte,rounds=1200,seeds=(11,22,33),lr=0.03,md=6,mcw=5):
    ps = dict(objective="reg:quantileerror",quantile_alpha=0.5,learning_rate=lr,max_depth=md,subsample=0.7,colsample_bytree=0.7,min_child_weight=mcw,reg_lambda=1.0,tree_method="hist",eval_metric="quantile")
    dtr=xgb.DMatrix(Xtr,label=ytr,weight=wtr); dte=xgb.DMatrix(Xte); pr=[]
    for s in seeds:
        ps["seed"]=s; b=xgb.train(ps,dtr,rounds); pr.append(b.predict(dte))
    return np.mean(pr,axis=0)

for featsX,name in [(feats,"noSD"),(feats2,"withSD")]:
    Xf = Ft[featsX].astype("float32")
    maes=[]
    for hold in [347,403,431]:
        tr=day<hold; te=day==hold
        w=0.5**((hold-day[tr])/140.0)
        p=run(Xf[tr],y[tr],w,Xf[te])
        maes.append(np.abs(p-y[te]).mean())
    print(name, [round(m,2) for m in maes], "avg", round(np.mean(maes),3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time, xgboost as xgb
f4 = A.load_saved("e004_features.parquet"); f5 = A.load_saved("e005_newfeats.parquet")
F = f4.merge(f5, on=["household_key","snapshot_day"], how="inner")
tt = A.train_targets(); Ft = F.merge(tt, on=["household_key","snapshot_day"])
feats = [c for c in F.columns if c not in ("household_key","snapshot_day")]
Ft["r28_364"]=Ft.spend_28/(Ft.spend_364/13+1e-6); Ft["r7_364"]=Ft.spend_7/(Ft.spend_364/52+1e-6)
Ft["r84_364"]=Ft.spend_84/(Ft.spend_364/4+1e-6); Ft["act_ratio_28_84"]=Ft.act_28/(Ft.act_84+1e-6)
feats += ["r28_364","r7_364","r84_364","act_ratio_28_84"]
X = Ft[feats].astype("float32"); y = Ft.future_spend_4w.values.astype("float32"); day = Ft.snapshot_day.values
ps = dict(objective="reg:quantileerror",quantile_alpha=0.5,learning_rate=0.03,max_depth=6,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,reg_lambda=1.0,tree_method="hist",eval_metric="quantile")
def run(Xtr,ytr,wtr,Xte,rounds=1200,seeds=(11,22,33)):
    dtr=xgb.DMatrix(Xtr,label=ytr,weight=wtr); dte=xgb.DMatrix(Xte); pr=[]
    for s in seeds:
        ps["seed"]=s; b=xgb.train(ps,dtr,rounds); pr.append(b.predict(dte))
    return np.mean(pr,axis=0)
hold=431
tr=day<hold; te=day==hold
w=0.5**((hold-day[tr])/140.0)
p=run(X[tr],y[tr],w,X[te])
r = y[te]-p
print("hold431 MAE", np.abs(r).mean().round(2), "bias(y-pred) mean", r.mean().round(2), "median", np.median(r).round(2))
# error by segment
dte = Ft[te].copy(); dte["p"]=p; dte["y"]=y[te]; dte["absr"]=np.abs(r)
dte["seg"] = pd.cut(dte.y, [-1,0.01,50,100,200,400,10000], labels=["0","0-50","50-100","100-200","200-400","400+"])
print(dte.groupby("seg",observed=True).agg(n=("absr","size"), mae=("absr","mean"), bias=("r","mean"), meanp=("p","mean"), meany=("y","mean")).round(1))
print("share of total MAE by seg:", (dte.groupby("seg",observed=True).absr.sum()/dte.absr.sum()).round(3))
# zero-target households: what does model predict?
z = dte[dte.y==0]
print("y=0: n", len(z), "mean pred", z.p.mean().round(1), "median pred", z.p.median().round(1), "their spend_28 mean", z.spend_28.mean().round(1))
# among y=0, mae by pred decile
print("MAE if we predicted 0 for all y=0 rows:", z.p.mean().round(2), "vs current", z.absr.mean().round(2))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb
f4 = A.load_saved("e004_features.parquet"); f5 = A.load_saved("e005_newfeats.parquet")
F = f4.merge(f5, on=["household_key","snapshot_day"], how="inner")
tt = A.train_targets(); Ft = F.merge(tt, on=["household_key","snapshot_day"])
feats = [c for c in F.columns if c not in ("household_key","snapshot_day")]
Ft["r28_364"]=Ft.spend_28/(Ft.spend_364/13+1e-6); Ft["r7_364"]=Ft.spend_7/(Ft.spend_364/50+1e-6)
Ft["r84_364"]=Ft.spend_84/(Ft.spend_364/4+1e-6); Ft["act_ratio_28_84"]=Ft.act_28/(Ft.act_84+1e-6)
feats += ["r28_364","r7_364","r84_364","act_ratio_28_84"]
X = Ft[feats].astype("float32"); y = Ft.future_spend_4w.values.astype("float32"); day = Ft.snapshot_day.values
ps = dict(objective="reg:quantileerror",quantile_alpha=0.5,learning_rate=0.03,max_depth=6,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,reg_lambda=1.0,tree_method="hist",eval_metric="quantile")
def run(Xtr,ytr,wtr,Xte,rounds=1200,seeds=(11,22,33)):
    dtr=xgb.DMatrix(Xtr,label=ytr,weight=wtr); dte=xgb.DMatrix(Xte); pr=[]
    for s in seeds:
        ps["seed"]=s; b=xgb.train(ps,dtr,rounds); pr.append(b.predict(dte))
    return np.mean(pr,axis=0)
hold=431; tr=day<hold; te=day==hold
w=0.5**((hold-day[tr])/140.0)
p=run(X[tr],y[tr],w,X[te])
dte = Ft[te].copy(); dte["p"]=p; dte["y"]=y[te]; dte["absr"]=np.abs(y[te]-p); dte["r"]=y[te]-p
print("hold431 MAE", dte.absr.mean().round(2))
dte["seg"] = pd.cut(dte.y, [-1,0.01,50,100,200,400,10000], labels=["0","0-50","50-100","100-200","200-400","400+"])
print(dte.groupby("seg",observed=True).agg(n=("absr","size"), mae=("absr","mean"), bias=("r","mean"), meanp=("p","mean"), meany=("y","mean")).round(1))
print("share of total abs error:", (dte.groupby("seg",observed=True).absr.sum()/dte.absr.sum()).round(3))
z = dte[dte.y==0]
print("y=0: n", len(z), "mean pred", round(z.p.mean(),1), "median pred", round(z.p.median(),1), "spend28 mean", round(z.spend_28.mean(),1))
print("pred quantiles for y=0:", np.percentile(z.p,[10,50,90]).round(1))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb
f4 = A.load_saved("e004_features.parquet"); f5 = A.load_saved("e005_newfeats.parquet")
F = f4.merge(f5, on=["household_key","snapshot_day"], how="inner")
tt = A.train_targets(); Ft = F.merge(tt, on=["household_key","snapshot_day"])
feats = [c for c in F.columns if c not in ("household_key","snapshot_day")]
Ft["r28_364"]=Ft.spend_28/(Ft.spend_364/13+1e-6); Ft["r7_364"]=Ft.spend_7/(Ft.spend_364/50+1e-6)
Ft["r84_364"]=Ft.spend_84/(Ft.spend_364/4+1e-6); Ft["act_ratio_28_84"]=Ft.act_28/(Ft.act_84+1e-6)
feats += ["r28_364","r7_364","r84_364","act_ratio_28_84"]
X = Ft[feats].astype("float32"); y = Ft.future_spend_4w.values.astype("float32"); day = Ft.snapshot_day.values
def train_one(obj,Xtr,ytr,wtr,rounds=1200,seeds=(11,22,33),lr=0.03,md=6):
    ps = dict(objective=obj,quantile_alpha=0.5,learning_rate=lr,max_depth=md,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,reg_lambda=1.0,tree_method="hist",eval_metric="quantile" if obj.startswith("reg:quantile") else "mae")
    dtr=xgb.DMatrix(Xtr,label=ytr,weight=wtr); pr=[]
    for s in seeds:
        ps["seed"]=s; b=xgb.train(ps,dtr,rounds); pr.append(b.predict(xgb.DMatrix(Xtr if Xte is None else Xte)))
    return np.mean(pr,axis=0)
def train_one(obj,Xtr,ytr,wtr,Xte,rounds=1200,seeds=(11,22,33),lr=0.03,md=6):
    ps = dict(objective=obj,quantile_alpha=0.5,learning_rate=lr,max_depth=md,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,reg_lambda=1.0,tree_method="hist",eval_metric="quantile" if "quantile" in obj else "mae")
    dtr=xgb.DMatrix(Xtr,label=ytr,weight=wtr); dte=xgb.DMatrix(Xte); pr=[]
    for s in seeds:
        ps["seed"]=s; b=xgb.train(ps,dtr,rounds); pr.append(b.predict(dte))
    return np.mean(pr,axis=0)

hold=431; tr=day<hold; te=day==hold
w = 0.5**((hold-day[tr])/140.0)
pq = train_one("reg:quantileerror",X[tr],y[tr],w,X[te])
pm = train_one("reg:squarederror",X[tr],y[tr],w,X[te])
yv=y[te]
print("MAE quantile", np.abs(pq-yv).mean().round(2), " MAE sqerr", np.abs(pm-yv).mean().round(2))
for wq in [0.8,0.7,0.6,0.5]:
    p = wq*pq+(1-wq)*pm
    print(f"blend wq={wq}: MAE", np.abs(p-yv).mean().round(2))
# log-target squared error with smearing
yl = np.log1p(y[tr])
pl = train_one("reg:squarederror",X[tr],yl,w,X[te])
resid = yl - None
# smearing factor from train
dtr=xgb.DMatrix(X[tr],label=yl,weight=w)
ps=dict(objective="reg:squarederror",learning_rate=0.03,max_depth=6,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,reg_lambda=1.0,tree_method="hist")
b=xgb.train(ps,dtr,1200); intr=b.predict(dtr)
fac = np.mean(np.exp(yl-intr))
plog = np.expm1(pl)*fac
print("MAE log-model smeared:", np.abs(plog-yv).mean().round(2))
for wq in [0.7,0.5]:
    p=wq*pq+(1-wq)*plog
    print(f"blend q+log wq={wq}: MAE", np.abs(p-yv).mean().round(2))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings("ignore")
f4 = A.load_saved("e004_features.parquet"); f5 = A.load_saved("e005_newfeats.parquet")
F = f4.merge(f5, on=["household_key","snapshot_day"], how="inner")
tt = A.train_targets(); Ft = F.merge(tt, on=["household_key","snapshot_day"])
feats = [c for c in F.columns if c not in ("household_key","snapshot_day")]
Ft["r28_364"]=Ft.spend_28/(Ft.spend_364/13+1e-6); Ft["r7_364"]=Ft.spend_7/(Ft.spend_364/50+1e-6)
Ft["r84_364"]=Ft.spend_84/(Ft.spend_364/4+1e-6); Ft["act_ratio_28_84"]=Ft.act_28/(Ft.act_84+1e-6)
feats += ["r28_364","r7_364","r84_364","act_ratio_28_84"]
X = Ft[feats].astype("float32"); y = Ft.future_spend_4w.values.astype("float32"); day = Ft.snapshot_day.values
def train_ps(Xtr,ytr,wtr,Xte,ps,rounds,seed=11):
    ps=dict(ps); ps["tree_method"]="hist"; ps["seed"]=seed
    b=xgb.train(ps,xgb.DMatrix(Xtr,label=ytr,weight=wtr),rounds)
    return b.predict(xgb.DMatrix(Xte))
hold=431; tr=day<hold; te=day==hold; yv=y[te]
w=0.5**((hold-day[tr])/140.0)
base=dict(objective="reg:quantileerror",quantile_alpha=0.5,learning_rate=0.03,max_depth=6,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,reg_lambda=1.0,eval_metric="quantile")
configs = {
 "d6 (ref)": {},
 "d8": dict(max_depth=8),
 "mcw20": dict(min_child_weight=20),
 "col0.5": dict(colsample_bytree=0.5),
 "lam5": dict(reg_lambda=5.0),
 "lr.05": dict(learning_rate=0.05),
 "sub.9": dict(subsample=0.9),
}
for name,ov in configs.items():
    ps={**base,**ov}
    p=np.mean([train_ps(X[tr],y[tr],w,X[te],ps,1200,s) for s in (11,22,33)],axis=0)
    print(f"{name:10s} MAE {np.abs(p-yv).mean():.2f}")
# log-target
yl=np.log1p(y[tr])
psl=dict(objective="reg:squarederror",learning_rate=0.03,max_depth=6,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,reg_lambda=1.0,tree_method="hist",seed=11)
b=xgb.train(psl,xgb.DMatrix(X[tr],label=yl,weight=w),1200)
intr=b.predict(xgb.DMatrix(X[tr])); fac=np.mean(np.exp(yl-intr))
pl=b.predict(xgb.DMatrix(X[te])); plog=np.expm1(pl)*fac
print("log-smeared MAE", np.abs(plog-yv).mean().round(2))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings("ignore")
f4 = A.load_saved("e004_features.parquet"); f5 = A.load_saved("e005_newfeats.parquet")
F = f4.merge(f5, on=["household_key","snapshot_day"], how="inner")
tt = A.train_targets(); Ft = F.merge(tt, on=["household_key","snapshot_day"])
feats = [c for c in F.columns if c not in ("household_key","snapshot_day")]
Ft["r28_364"]=Ft.spend_28/(Ft.spend_364/13+1e-6); Ft["r7_364"]=Ft.spend_7/(Ft.spend_364/50+1e-6)
Ft["r84_364"]=Ft.spend_84/(Ft.spend_364/4+1e-6); Ft["act_ratio_28_84"]=Ft.act_28/(Ft.act_84+1e-6)
feats += ["r28_364","r7_364","r84_364","act_ratio_28_84"]
X = Ft[feats].astype("float32"); y = Ft.future_spend_4w.values.astype("float32"); day = Ft.snapshot_day.values
base=dict(objective="reg:quantileerror",quantile_alpha=0.5,learning_rate=0.03,max_depth=6,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,reg_lambda=1.0,eval_metric="quantile",tree_method="hist")
def pred(ps,Xtr,ytr,wtr,Xte,rounds=1200,seeds=(11,22,33)):
    ps=dict(ps); pr=[]
    for s in seeds:
        ps["seed"]=s; b=xgb.train(ps,xgb.DMatrix(Xtr,label=ytr,weight=wtr),rounds); pr.append(b.predict(xgb.DMatrix(Xte)))
    return np.mean(pr,axis=0)
# ref model on 431 and on 403 (for calibration fit)
tr=day<431; te=day==431; w=0.5**((431-day[tr])/140.0)
p431 = pred(base,X[tr],y[tr],w,X[te]); yv=y[te]
tr2=day<403; te2=day==403; w2=0.5**((403-day[tr2])/140.0)
p403 = pred(base,X[tr2],y[tr2],w2,X[te2]); y403=y[te2]
print("ref 431 MAE", np.abs(p431-yv).mean().round(2), "| 403 MAE", np.abs(p403-y403).mean().round(2))
# calibration: linear y ~ p fit on day-403 preds, applied to 431 preds
b,a = np.polyfit(p403,y403,1)
print("calib slope",round(b,3),"intercept",round(a,1))
pc = b*p431+a
print("linear-calib 431 MAE", np.abs(pc-yv).mean().round(2), " mean pred", pc.mean().round(1))
# isotonic
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds="clip").fit(p403,y403)
pi = iso.predict(p431)
print("isotonic-calib 431 MAE", np.abs(pi-yv).mean().round(2), " mean pred", pi.mean().round(1))
# regularized variant
ps20 = {**base,"min_child_weight":20}
pr20 = pred(ps20,X[tr],y[tr],w,X[te])
print("mcw20 431 MAE", np.abs(pr20-yv).mean().round(2))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, warnings, time
warnings.filterwarnings("ignore")
f4 = A.load_saved("e004_features.parquet"); f5 = A.load_saved("e005_newfeats.parquet")
F = f4.merge(f5, on=["household_key","snapshot_day"], how="inner")
tt = A.train_targets(); Ft = F.merge(tt, on=["household_key","snapshot_day"])
feats = [c for c in F.columns if c not in ("household_key","snapshot_day")]
Ft["r28_364"]=Ft.spend_28/(Ft.spend_364/13+1e-6); Ft["r7_364"]=Ft.spend_7/(Ft.spend_364/50+1e-6)
Ft["r84_364"]=Ft.spend_84/(Ft.spend_364/4+1e-6); Ft["act_ratio_28_84"]=Ft.act_28/(Ft.act_84+1e-6)
feats += ["r28_364","r7_364","r84_364","act_ratio_28_84"]
X = Ft[feats].astype("float32"); y = Ft.future_spend_4w.values.astype("float32"); day = Ft.snapshot_day.values
t0=time.time()
def pred(ps,Xtr,ytr,wtr,Xte,rounds=1200,seeds=(11,22,33)):
    ps=dict(ps); pr=[]
    for s in seeds:
        ps["seed"]=s; b=xgb.train(ps,xgb.DMatrix(Xtr,label=ytr,weight=wtr),rounds); pr.append(b.predict(xgb.DMatrix(Xte)))
    return np.mean(pr,axis=0)
hold=431; tr=day<hold; te=day==hold; yv=y[te]
w=0.5**((hold-day[tr])/140.0)
base=dict(learning_rate=0.03,max_depth=6,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,reg_lambda=1.0,tree_method="hist")
pq = pred({**base,"objective":"reg:quantileerror","quantile_alpha":0.5,"eval_metric":"quantile"},X[tr],y[tr],w,X[te])
pb = pred({**base,"objective":"binary:logistic","eval_metric":"logloss"},X[tr],(y[tr]>0).astype(int),w,X[te])
pos = y[tr]>0
pp = pred({**base,"objective":"reg:quantileerror","quantile_alpha":0.5,"eval_metric":"quantile"},X[tr][pos],y[tr][pos],w[pos],X[te])
two = np.where(pb>0.5, 0.0, pp)
print(f"t={time.time()-t0:.0f}s quantile {np.abs(pq-yv).mean():.2f} | two-part {np.abs(two-yv).mean():.2f} | blend .5 {np.abs((0.5*pq+0.5*two)-yv).mean():.2f}")
print("P(0) mean", pb.mean().round(3), "actual zerofrac", (yv==0).mean().round(3))
# log model
yl=np.log1p(y[tr])
pl = pred({**base,"objective":"reg:squarederror","eval_metric":"mae"},X[tr],yl,w,X[te])
b2=xgb.train({**base,"objective":"reg:squarederror","seed":11},xgb.DMatrix(X[tr],label=yl,weight=w),1200)
fac=np.mean(np.exp(yl-b2.predict(xgb.DMatrix(X[tr]))))
plog=np.expm1(pl)*fac
print("log-smeared", np.abs(plog-yv).mean().round(2), "| blend q+log .7/.3", np.abs((0.7*pq+0.3*plog)-yv).mean().round(2), "| q+log+2p", np.abs((0.6*pq+0.2*plog+0.2*two)-yv).mean().round(2))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, warnings, time
warnings.filterwarnings("ignore")
f4 = A.load_saved("e004_features.parquet"); f5 = A.load_saved("e005_newfeats.parquet")
F = f4.merge(f5, on=["household_key","snapshot_day"], how="inner")
feats = [c for c in F.columns if c not in ("household_key","snapshot_day")]
F["r28_364"]=F.spend_28/(F.spend_364/13+1e-6); F["r7_364"]=F.spend_7/(F.spend_364/50+1e-6)
F["r84_364"]=F.spend_84/(F.spend_364/4+1e-6); F["act_ratio_28_84"]=F.act_28/(F.act_84+1e-6)
feats += ["r28_364","r7_364","r84_364","act_ratio_28_84"]
tr_rows = F.snapshot_day<=431; va_rows = F.snapshot_day>=459
Xtr = F.loc[tr_rows, feats].astype("float32").values
w = 0.5**((459 - F.loc[tr_rows,"snapshot_day"].values)/140.0)
tt = A.train_targets(); ytr = F.loc[tr_rows,["household_key","snapshot_day"]].merge(tt, on=["household_key","snapshot_day"]).future_spend_4w.values.astype("float32")
assert len(ytr)==tr_rows.sum()
Xva = F.loc[va_rows, feats].astype("float32").values
ps = dict(objective="reg:quantileerror",quantile_alpha=0.5,learning_rate=0.03,max_depth=6,subsample=0.7,colsample_bytree=0.7,min_child_weight=5,reg_lambda=1.0,tree_method="hist",eval_metric="quantile")
dtr=xgb.DMatrix(Xtr,label=ytr,weight=w); dva=xgb.DMatrix(Xva)
t0=time.time(); pr=[]
for s in (11,22,33,44,55):
    ps["seed"]=s; b=xgb.train(ps,dtr,2400); pr.append(b.predict(dva))
    print("seed",s,"done",round(time.time()-t0))
pred = np.mean(pr,axis=0)
out = F.loc[va_rows,["household_key","snapshot_day"]].copy(); out["prediction"]=pred
print(out.shape, out.prediction.describe().round(1).to_dict())
p = A.save_table(out, "e007_preds")
print("path:", p)
