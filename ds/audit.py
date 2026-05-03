import polars as pl

pl.Config.set_tbl_rows(-1) #max rows shown
pl.Config.set_tbl_cols(-1) #max cols shown
#pl.Config.set_tbl_rows(-1)
#pl.Config.set_tbl_rows(-1)


PATH = "./ds/DCLDE_w_Buzzes.csv"
SCHEMA = {
    "Soundfile": pl.String,
    "Dataset": pl.String,
    "LowFreqHz": pl.Float32,
    "HighFreqHz": pl.Float32,
    "FileEndSec": pl.Float32,
    "UTC": pl.Datetime,
    "FileBeginSec": pl.Float32,
    "ClassSpecies": pl.String,
    "KW": pl.Boolean, # need to replace strict "FALSE" to python's False
    "KW_certain": pl.Boolean,
    "Ecotype": pl.String, 
    "Provider": pl.String, 
    "AnnotationLevel": pl.String,
    "FilePath": pl.String, 
    "FileOk": pl.Boolean, 
    "CallType": pl.String, 
    "ID": pl.Int32, #fails for two values: ["2e+05", "1e+05"]
    "CenterTime": pl.Float64,
    "Duration": pl.Float64,
    "CalltypeCategory": pl.String,
    "HasQ": pl.Boolean, 
    "CalltypeHasQ": pl.Boolean, 
    "EcotypeCertain": pl.Boolean,
    "isPulsedCalls": pl.Boolean, 
    "Labels": pl.String, 
    "HoldOut": pl.String,
    "Buzz": pl.Boolean
}

def load_raw():
    return pl.read_csv(PATH, infer_schema_length=0, null_values=["NA", "", "NaN", "null", "None"])

def column_summary(df:pl.DataFrame):
    #Inspect the df per column
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "col": col,
            "n_unique": s.n_unique(),
            "n_null": s.null_count(),
            "samples": s.drop_nulls().unique().head(5).to_list(),

        })
    return pl.DataFrame(rows)

def cast_failures(df:pl.DataFrame):
    exprs = []
    for c, t in SCHEMA.items():
        col = pl.col(c)
        if t in (pl.Datetime, pl.Date):
            parsed = pl.col(c).str.strptime(t,strict=True)
        elif t == pl.Boolean:
            parsed = pl.col(c).replace_strict({"FALSE": False, "TRUE": True, "1": True, "0": False}).cast(t, strict=False)
        else:
            parsed = pl.col(c).cast(t, strict=False)
        exprs.append((parsed.is_null() & col.is_not_null()).sum().alias(f"{c}_fail"))
    return df.select(exprs)


def load_clean():
    df_clean = load_raw()
    df_clean = df_clean.with_columns(pl.col('ID').replace({"2e+05": "200000", "1e+05": "100000"}))
    for c, t in SCHEMA.items():
        if t == pl.Datetime:
            df_clean = df_clean.with_columns(pl.col(c).str.strptime(t,strict=True).cast(t, strict=True))
        elif t == pl.Boolean:
            df_clean = df_clean.with_columns(pl.col(c).replace_strict({"FALSE": False, "TRUE": True, "1": True, "0": False}).cast(t, strict=True))
        else:
            df_clean = df_clean.with_columns(pl.col(c).cast(t, strict=True))
    return df_clean

if __name__ == "__main__":
    df_clean = load_clean()
    df_clean.write_parquet("./ds/DCLDE_w_Buzzes.parquet")










