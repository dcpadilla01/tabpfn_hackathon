df = agent_api.load_saved("e013_stationary.parquet")
tgt = agent_api.train_targets()
v = agent_api.snapshot()
dm = v.display_mailer.copy()
tx = v.transactions.copy()
dm["disp_any"] = dm.display.astype(str).ne("0").astype(int)
dm["mail_any"] = dm.mailer.astype(str).ne("0").astype(int)
txw = tx.merge(dm[["product_id","store_id","week_no","disp_any","mail_any"]], on=["product_id","store_id","week_no"], how="inner")
print("matched tx rows:", len(txw), "hh:", txw.household_key.nunique())

g = txw.groupby("household_key").agg(
    exp_wk=("week_no","nunique"),
    disp_rows=("disp_any","sum"),
    mail_rows=("mail_any","sum"),
)
m = df.merge(tgt, on=["household_key","snapshot_day"], how="left")
mm = m[m.snapshot_day==459][["household_key","future_spend_4w"]].merge(g, left_on="household_key", right_index=True, how="left").fillna({"exp_wk":0,"disp_rows":0,"mail_rows":0})
print(mm[["future_spend_4w","exp_wk","disp_rows","mail_rows"]].corr().round(3))
