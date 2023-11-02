"""World Ozone and Ultraviolet Radiation Data Centre,
including database of ozonesondes.

Primary website: https://woudc.org/data/dataset_info.php?id=ozonesonde

Data access: https://woudc.org/archive/

"""
import io
import warnings

import pandas as pd


def get_archive(dates=None, which="ozonesonde"):
    """Load WOUDC archive data.

    From https://woudc.org/archive/Summaries/dataset-snapshots

    Parameters
    ----------
    dates : array-like of datetime-like, optional
        Dates corresponding to the desired period of interest.
        Default: return full archive.
    which : {'ozonesonde', 'totalozone', 'totalozoneobs'}
        Which data archive to get.

    Returns
    -------
    pandas.DataFrame
        'data_block's expanded if present.
    """
    base_url = "https://woudc.org/archive/Summaries/dataset-snapshots/"

    url = f"{base_url}{which}.zip"

    print(f"Loading WOUDC {which} archive...")
    df = pd.read_csv(url, dtype=str, delimiter=",", engine="c", low_memory=False)

    if which == "ozonesonde":
        # NOTE: 2.1 GB zipped, takes a while to load
        # NOTE: 'flight_plot_path' has URL to ozonesonde profile plot if available
        float_cols = [
            "flight_summary_integratedo3",
            "flight_summary_sondetotalo3",
            "flight_summary_correctionfactor",
            "flight_summary_totalo3",
        ]
        int_cols = []
        bool_cols = ["latest_observation"]
        date_cols = ["instance_datetime", "generation_datetime"]
        data_block_pref = "ozonesonde_"
    elif which == "totalozone":
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
        data_block_pref = None
    elif which == "totalozoneobs":
        float_cols = ["X", "Y", "daily_summary_meano3", "daily_summary_stddevo3"]
        int_cols = ["daily_summary_nobs"]
        bool_cols = ["latest_observation"]  # seems to be all 0 and 1
        date_cols = ["instance_datetime"]
        data_block_pref = "ozone_"
    else:
        raise ValueError(f"WOUDC archive ID {which!r} invalid or not implemented.")

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

    if dates is not None:
        df = df[df.instance_datetime.between(dates.min(), dates.max(), inclusive="both")]

    if "data_block" in df.columns:
        # totalozoneobs has a column of CSV strings ('data_block')
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

        # ozonesonde 'data_block' columns:
        # - Pressure
        # - O3PartialPressure
        # - Temperature
        # - WindSpeed
        # - WindDirection
        # - LevelCode
        # - Duration
        # - GPHeight
        # - RelativeHumidity
        # - SampleTemperature

        print("Expanding data blocks...")
        data = []
        for i in range(len(df)):
            # TODO: dtypes (e.g. 'time' -> timedelta)
            data_i = pd.read_csv(io.StringIO(df.data_block.iloc[i]))
            data_i["data_payload_id"] = df.data_payload_id.iloc[i]
            data.append(data_i)
        data = pd.concat(data, ignore_index=True)
        data = data.rename(
            columns=lambda col: f"{data_block_pref}{col.lower()}"
            if col != "data_payload_id"
            else col
        )

        df = data.merge(df.drop(columns=["data_block"]), how="left", on="data_payload_id")

    return df


def get_available_dates(*, n_threads=10):
    """Of individual ozonesonde files, in
    https://woudc.org/archive/Archive-NewFormat/OzoneSonde_1.0_1/
    """
    import re
    from multiprocessing.pool import ThreadPool

    import requests

    url = "https://woudc.org/archive/Archive-NewFormat/OzoneSonde_1.0_1/"

    # Get subdirectories (sites)
    r = requests.get(url)
    r.raise_for_status()
    text = r.text
    sites = re.findall(r'alt="\[DIR\]"></td><td><a href="([^"]+)/">\1/</a>', text)

    # Discover CSV directories
    def get_dirs(site):
        r = requests.get(url + site + "/")
        r.raise_for_status()
        text = r.text.replace("%20", " ")
        group_dirs = re.findall(r'alt="\[DIR\]"></td><td><a href="([^"]+)/">\1/</a>', text)
        if not group_dirs:
            warnings.warn(f"No group directories found for {site} ({r.url}).")

        dirs = []
        for subdir in group_dirs:
            r = requests.get(url + site + "/" + subdir + "/")
            r.raise_for_status()
            text = r.text  # .replace("%20", " ")
            year_dirs = re.findall(r'alt="\[DIR\]"></td><td><a href="([^"]+)/">\1/</a>', text)
            if not year_dirs:
                warnings.warn(f"No year directories found for {site}/{subdir} ({r.url}).")
            dirs.extend(r.url + year + "/" for year in year_dirs)

        return dirs

    pool = ThreadPool(processes=n_threads)
    dirs = list(pool.imap_unordered(get_dirs, sites))
    print(dirs)

    # Discover CSV files

    # for year in year_dirs:
    #     print(year)
    #     r = requests.get(url + site + "/" + subdir + "/" + year + "/")
    #     r.raise_for_status()
    #     text = r.text.replace("%20", " ")
    #     files = re.findall(r'href="([^"]+\.csv)">\1</a>', text)
    #     print(files)


get_available_dates()
