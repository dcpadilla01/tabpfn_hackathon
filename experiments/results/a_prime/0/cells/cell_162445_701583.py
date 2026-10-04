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
