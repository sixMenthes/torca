import polars as pl
import gcsfs
import soundfile as sf

pl.Config.set_tbl_rows(-1) #max rows shown
pl.Config.set_tbl_cols(-1) #max cols shown
pl.Config.set_fmt_str_lengths(100)
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

GCL = gcsfs.core.GCSFileSystem(token="anon")

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
    df_clean = df_clean.with_columns(NewPath = pl.col("FilePath").map_elements(mod_dl_path, return_dtype=pl.Utf8))
    return df_clean

def mod_dl_path(old_path):
    new_root = "gs://noaa-passive-bioacoustic/dclde/2027/dclde_2027_killer_whales/"
    file_name = old_path.split("/")[-1]
    parts_path = [s.lower() for s in old_path.split("/")[2:-1]]
    new_path = "/".join(parts_path) + "/" + file_name
    new_path = new_root + new_path
    return new_path

def check_data_exists(df_clean):
    # mod_dl_path's reconstruction misses files whose bucket path renames a
    # directory (Quin_Can→qc, Lime Kiln→lime-kiln, StraitofGeorgia→
    # straitofgeorgia_globus-robertsbank, …) or whose CSV FilePath contains a
    # stray "//". Audio basenames in the bucket are unique, so resolve by
    # basename and overwrite NewPath with the actual cloud location.
    gclient = GCL
    root = "noaa-passive-bioacoustic/dclde/2027/dclde_2027_killer_whales/"
    files = gclient.find(root)
    by_basename = {p.rsplit("/", 1)[-1]: p for p in files}

    df_clean = df_clean.with_columns(
        BaseName = pl.col("FilePath").str.split("/").list.last(),
    )
    df_clean = df_clean.with_columns(
        ResolvedPath = pl.col("BaseName").replace_strict(
            by_basename, default=None, return_dtype=pl.Utf8
        ),
    )
    df_clean = df_clean.with_columns(
        NewFileOk = pl.col("ResolvedPath").is_not_null(),
        NewPath = pl.when(pl.col("ResolvedPath").is_not_null())
                    .then("gs://" + pl.col("ResolvedPath"))
                    .otherwise(pl.col("NewPath")),
    ).drop("BaseName", "ResolvedPath")

    diff = df_clean.filter(pl.col("FileOk") & (~pl.col("NewFileOk")))
    missing_counts = diff.group_by("Provider", "Dataset").agg(
        pl.col("NewPath").n_unique().alias("n_missing")
    ).sort("n_missing", descending=True)
    missing_ds = diff.select("FilePath", "Provider", "Dataset", "NewPath")
    return df_clean, missing_counts, missing_ds



def check_file_exists(file_path):
    gclient = GCL
    root = "noaa-passive-bioacoustic/dclde/2027/dclde_2027_killer_whales/"
    file_path = file_path.removeprefix("gs://")
    files = set(gclient.find(root))
    return file_path in files

def add_file_durations(df_clean):
    # Each row is a slice; many slices share a file. Fetch duration once per
    # unique resolved path by reading the audio header from GCS, then join.
    gclient = GCL
    paths = (
        df_clean.filter(pl.col("NewFileOk"))
        .select("NewPath")
        .unique()
        .to_series()
        .to_list()
    )
    durations = {}
    for p in paths:
        try:
            with gclient.open(p.removeprefix("gs://"), "rb") as f:
                info = sf.info(f)
            durations[p] = info.frames / info.samplerate
        except Exception:
            durations[p] = None

    return df_clean.with_columns(
        FileLen = pl.col("NewPath").replace_strict(
            durations, default=None, return_dtype=pl.Float64
        ),
    )


if __name__ == "__main__":
    df_clean = load_clean()
    df_clean, missing_counts, missing_ds = check_data_exists(df_clean)
    missing_ds.write_csv("./ds/missing_dclde.csv")
    df_clean = add_file_durations(df_clean)
    df_clean.write_parquet("./ds/DCLDE_w_Buzzes.parquet")










