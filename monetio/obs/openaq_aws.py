"""OpenAQ archive data on AWS.

https://openaq.org/

https://registry.opendata.aws/openaq/

https://docs.openaq.org/docs/accessing-openaq-archive-data
"""

import logging
import warnings

import pandas as pd

logger = logging.getLogger(__name__)


def read(fp):
    """Read OpenAQ archive data from a file-like object.

    Parameters
    ----------
    fp : str or path-like or file-like
        OpenAQ archive data, suitable for passing to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
        OpenAQ archive data.
    """

    df = pd.read_csv(
        fp,
        dtype={
            0: str,  # location_id
            1: str,  # sensor_id or sensors_id ??
            2: str,  # location
            # 3: datetime
            4: float,  # lat
            5: float,  # lon
            6: str,  # parameter
            7: str,  # unit or units ??
            8: float,  # value
        },
        parse_dates=["datetime"],
    )

    # Normalize to web API column names
    if "sensors_id" in df.columns:
        df = df.rename(columns={"sensors_id": "sensor_id"})
    if "units" in df.columns:
        df = df.rename(columns={"units": "unit"})
    df = df.rename(
        columns={
            "location_id": "siteid",
            "datetime": "time",
            "lat": "latitude",
            "lon": "longitude",
        }
    )

    # Convert to UTC, non-localized
    df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)

    return df


def get_paths(dates, *, location_ids=None):
    """
    Parameters
    ----------
    dates : pd.DatetimeIndex
    """
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)

    if location_ids is None:
        warnings.warn(
            "location ID(s) not provided; using all locations, which may be quite slow",
            stacklevel=2,
        )
        location_ids = ["*"]

    tpl = (
        "openaq-data-archive/records/csv.gz/"
        "locationid={loc}/year={date:%Y}/month={date:%m}/"
        "location-{loc}-{date:%Y%m%d}.csv.gz"
    )

    print("discovering paths...")
    paths = []
    for date in dates.floor("D").unique():
        for loc in location_ids:
            glb = tpl.format(loc=loc, date=date)
            if "*" in glb:
                loc_date_paths = fs.glob(glb)
                logger.debug(f"found {len(loc_date_paths)} path(s) for glob='{glb}'")
                paths.extend(loc_date_paths)
            else:
                if fs.exists(glb):
                    logger.debug(f"path exists: {glb}")
                    paths.append(glb)
                else:
                    logger.debug(f"path does not exist: {glb}")

    return paths


def add_data(
    dates,
    *,
    location_id=None,
    country=None,
    provider=None,
    n_procs=1,
):
    """Add OpenAQ data AWS Open Data.

    Parameters
    ----------
    dates : datetime-like or array-like of datetime-like
        Desired dates (the archive data is stored in daily files, per location).
    location_id : str or int or list, optional
        Location ID(s) to include.
    country : str or list of str, optional
        Country or countries to include.
    provider : str or list of str, optional
        Data provider(s) to include.

    Returns
    -------
    pd.DataFrame
        OpenAQ data.
    """
    import dask.dataframe as dd

    dates = pd.to_datetime(dates)
    if pd.api.types.is_scalar(dates):
        dates = pd.DatetimeIndex([dates])
    dates = dates.dropna()
    if dates.empty:
        raise ValueError("must provide at least one datetime-like")

    if pd.api.types.is_scalar(location_id):
        location_ids = [location_id]
    else:
        location_ids = location_id

    if country is not None or provider is not None:
        raise NotImplementedError("selection by location/country/provider not yet implemented")

    paths = get_paths(dates, location_ids=location_ids)
    print(f"found {len(paths)}")
    uris = [f"s3://{p}" for p in paths]

    meta = [
        ("siteid", str),
        ("sensor_id", str),
        ("location", str),
        ("time", "datetime64[ns]"),
        ("latitude", float),
        ("longitude", float),
        ("parameter", str),
        ("unit", str),
        ("value", float),
    ]
    print("reading...")
    df = dd.from_map(read, uris, meta=meta).compute(num_workers=n_procs)

    return df.reset_index(drop=True)
