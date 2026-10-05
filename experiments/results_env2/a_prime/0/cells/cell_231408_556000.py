import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

te = A.load_saved("e007_te.parquet")
tt = A.train_targets()
df = tt.merge(te, on=["household_key","snapshot_day"], how="left")
y = df[A.TARGET].values
def mae(p): return float(np.mean(np.abs(y-p)))

# ---- leak-free household outcome history (median / p_zero / ewm) ----
v = A.snapshot(459)  # capped view; enough to compute outcomes for windows ending <= 459
tr_days = A.snapshot_days()["train"]
tx = v.table("transactions")
sp = tx.groupby(["household_key","day"])["sales_value"].sum().reset_index()
# outcome for snapshot s' = spend in (s', s'+28]
def outcome_at(sp_end):
    m = (sp["day"] > sp_end-28) & (sp["day"] <= sp_end)
    g = sp[m].groupby("household_key")["sales_value"].sum()
    return g
outs = {}  # snapshot_day -> series of outcomes (only for train snapshot days whose window ends <= 459)
for s in tr_days:
    if s+28 <= 459:
        outs[s] = outcome_at(s+28)
days_sorted = sorted(outs)
# per household build history lists
hh_hist = {}
for s in days_sorted:
    o = outs[s]
    for h, val in o.items():
        hh_hist.setdefault(h, []).append((s, float(val)))

rows = []
for h, s in zip(df["household_key"], df["snapshot_day"]):
    hist = hh_hist.get(h, [])
    past = [val for (d, val) in hist if d < s]
    if past:
        med = float(np.median(past)); mean = float(np.mean(past))
        pz = float(np.mean([x==0 for x in past]))
        w = np.array([0.7**(len(past)-1-i) for i in range(len(past))])
        ewm = float(np.average(past, weights=w))
    else:
        med = mean = ewm = np.nan; pz = np.nan
    rows.append((med, mean, pz, ewm))
arr = np.array(rows, dtype=float)
df["te_med"] = arr[:,0]; df["te_mean_chk"] = arr[:,1]; df["p_zero"] = arr[:,2]; df["te_ewm"] = arr[:,3]
print("corr te_mean_chk vs te_hh_mean:", round(float(np.corrcoef(df["te_mean_chk"].fillna(-1), df["te_hh_mean"].fillna(-1))[0,1]),4))

med_g = float(np.median(y))
print("\nsingle-predictor MAE (train rows):")
print("te_hh_mean (E007):", round(mae(df["te_hh_mean"].fillna(df["spend_4w_lag1"]).fillna(med_g)),2))
print("te_med  (median):", round(mae(df["te_med"].fillna(df["spend_4w_lag1"]).fillna(med_g)),2))
print("te_ewm:", round(mae(df["te_ewm"].fillna(df["spend_4w_lag1"]).fillna(med_g)),2))
# two-part style: (1-p_zero)*median_pos
pos_med = df["te_med"].where(df["p_zero"]<1)  # median over positive outcomes
# compute median of positive past outcomes
rows2=[]
for h, s in zip(df["household_key"], df["snapshot_day"]):
    past=[val for (d,val) in hh_hist.get(h,[]) if d<s and val>0]
    rows2.append(float(np.median(past)) if past else np.nan)
df["te_medpos"]=np.array(rows2)
p2 = (1-df["p_zero"].fillna(0.207))*df["te_medpos"].fillna(med_g)
print("two-part (1-pz)*medpos:", round(mae(p2.values),2))
for w in [0.25,0.5,0.75]:
    print(f"blend two-part/te_med w={w}:", round(mae(w*p2.values+(1-w)*df["te_med"].fillna(med_g).values),2))

# ---- cohort drift ----
coh = df.groupby("snapshot_day").agg(trail=("spend_4w","mean"), ymean=(A.TARGET,"mean"), ymed=(A.TARGET,"median"))
print("\ncohort per snapshot:\n", coh.round(1))
print("corr trailing cohort spend_4w vs next-window target mean:", round(float(np.corrcoef(coh["trail"][:-1], coh["ymean"][1:])[0,1]),3))
