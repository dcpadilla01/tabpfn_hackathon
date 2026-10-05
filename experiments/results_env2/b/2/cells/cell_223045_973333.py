
import agent_api as A
print(A.snapshot_days())
for name in ["season_demo_v1","mkt_v1","rfm_cadence_v1","rfm_v1","demo_v1","ewma_block_v1","comp_v1","rfm_traj_v1"]:
    try:
        df = A.load_saved(name + ".parquet")
        cols = list(df.columns)
        print(name, df.shape, "nfeat=", len(cols)-2)
        print(cols[:60])
        print(cols[60:130])
        print(cols[130:])
    except Exception as e:
        print(name, "ERR", type(e).__name__, e)
v = A.snapshot()
t = v.transactions
print(t[['sales_value','coupon_disc','coupon_match_disc','retail_disc','quantity','trans_time']].describe())
r = v.table('coupon_redemptions')
print("redemptions", r.shape)
print(r.head())
