"""World Ozone and Ultraviolet Radiation Data Centre,
including database of ozonesondes.

Primary website: https://woudc.org/data/dataset_info.php?id=ozonesonde

Data access: https://woudc.org/archive/

"""
import pandas as pd

# Archive files
base_url = "https://woudc.org/archive/Summaries/dataset-snapshots/"

which = "totalozone"
url = f"{base_url}{which}.zip"

print(f"Loading WOUDC {which} archive...")
df = pd.read_csv(url, dtype=str, delimiter=",", engine="c", low_memory=False)

if which == "totalozone":
    float_cols = [
        "X",
        "Y",
        "monthly_columno3",
        "monthly_stddevo3",
        "daily_columno3",
        "daily_stddevo3",
        "daily_utc_begin",
        "daily_utc_end",
        "daily_utc_mean",
        "daily_mmu",
        "daily_columnso2",
    ]
    int_cols = ["monthly_npts", "daily_nobs"]
    bool_cols = ["latest_observation"]
    date_cols = ["instance_datetime", "monthly_date", "daily_date"]
elif which == "totalozoneobs":
    float_cols = ["X", "Y", "daily_summary_meano3", "daily_summary_stddevo3"]
    int_cols = ["daily_summary_nobs"]
    bool_cols = ["latest_observation"]  # seems to be all 0 and 1
    date_cols = [
        "instance_datetime",
    ]

print("Converting dtypes...")
# NOTE: It seems that the float cols are not clean, so simple `astype` doesn't work.
for col in float_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
for col in int_cols:
    # NOTE: generally can't be int dtype since missing values
    df[col] = pd.to_numeric(df[col], errors="coerce")
for col in bool_cols:
    df[col] = df[col].astype(bool)
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")
    if df[col].dt.tz is not None:
        df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)

df = df.rename(columns={"X": "longitude", "Y": "latitude", "gaw_id": "siteid"})

# totalozoneobs has a column of CSV strings ('data_block')
# Could process _after_ selecting time period of interest using 'instance_datetime'(?)
# - Time  (without date)
# - WLcode
# - ObsCode
# - Airmass
# - ColumnO3
# - StdDevO3
# - ColumnSO2
# - StdDevSO2
# - ZA
# - NdFilter
# - TempC
# - F324
