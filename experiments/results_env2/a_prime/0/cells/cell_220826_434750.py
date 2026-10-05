import agent_api as A
for name in ["e001_rfm","e003_union","e005_marketing"]:
    df = A.load_saved(name+".parquet")
    print(name, df.shape)
    print(list(df.columns))
    print()
