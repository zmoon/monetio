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


def _maybe_to_list(x):
    if x is not None and pd.api.types.is_scalar(x):
        return [x]
    else:
        return x


def get_paths(dates, *, location_id=None, provider=None):
    """
    Parameters
    ----------
    dates : pd.DatetimeIndex
    """
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)

    location_ids = _maybe_to_list(location_id)
    providers = _maybe_to_list(provider)

    if location_ids is None and providers is None:
        warnings.warn(
            "location ID(s) not provided; using all locations, which may be quite slow",
            stacklevel=2,
        )
        location_ids = ["*"]

    unique_dates = dates.floor("D").unique()

    print("discovering paths...")
    paths = []

    if location_ids is not None:
        tpl = (
            "openaq-data-archive/records/csv.gz/"
            "locationid={loc}/year={date:%Y}/month={date:%m}/"
            "location-{loc}-{date:%Y%m%d}.csv.gz"
        )
        for date in unique_dates:
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

    if providers is not None:
        tpl = (
            "openaq-data-archive/records/csv.gz/"
            "provider={prvdr}/country={cntry}/locationid={loc}/"
            "year={date:%Y}/month={date:%m}/"
            "location-{loc}-{date:%Y%m%d}.csv.gz"
        )
        for date in unique_dates:
            for prvdr in providers:
                glb = tpl.format(prvdr=prvdr.lower(), cntry="*", loc="*", date=date)
                print(glb)
                prvdr_date_paths = fs.glob(glb)
                logger.debug(f"found {len(prvdr_date_paths)} path(s) for glob='{glb}'")
                paths.extend(prvdr_date_paths)

    return sorted(set(paths))


def get_locs(*, country=None, provider=None):
    """Get location IDs corresponding to country/countries OR provider(s).

    Returns
    -------
    list of str
    """
    import re

    import s3fs

    if country is not None and provider is not None:
        raise ValueError("cannot specify both `country` and `provider`")

    print("discovering locations...")
    fs = s3fs.S3FileSystem(anon=True)
    if country is not None:
        if pd.api.types.is_scalar(country):
            countries = [country]
        else:
            countries = country

        paths = []
        for cntry in countries:
            cntry_paths = fs.find(
                f"openaq-data-archive/records/csv.gz/country={cntry.lower()}/",
                withdirs=True,
                maxdepth=1,
            )
            paths.extend(cntry_paths)

        locs = []
        for p in paths:
            m = re.fullmatch(
                r"openaq-data-archive/records/csv\.gz/country=([a-z]{2}|\-\-|99)/"
                r"locationid=([0-9]+)",
                p,
            )
            if m is not None:
                locs.append(m.group(2))

    elif provider is not None:
        if pd.api.types.is_scalar(provider):
            providers = [provider]
        else:
            providers = provider

        paths = []
        for prvdr in providers:
            prvdr_paths = fs.find(
                f"openaq-data-archive/records/csv.gz/provider={prvdr.lower()}/",
                withdirs=True,
                maxdepth=2,
            )
            paths.extend(prvdr_paths)

        locs = []
        for p in paths:
            m = re.fullmatch(
                r"openaq-data-archive/records/csv\.gz/provider=([a-z0-9\-]+)/"
                r"country=([a-z]{2}|\-\-|99)/locationid=([0-9]+)",
                p,
            )
            if m is not None:
                locs.append(m.group(3))

    else:  # All locs
        paths = fs.find("openaq-data-archive/records/csv.gz/", withdirs=True, maxdepth=1)
        locs = []
        for p in paths:
            m = re.fullmatch(r"openaq-data-archive/records/csv\.gz/locationid=([0-9]+)", p)
            if m is not None:
                locs.append(m.group(1))

    if not locs:
        warnings.warn(f"no locations found for country={country!r} provider={provider!r}")

    return sorted(locs)


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

    if country is not None:
        raise NotImplementedError("selection by country not yet implemented")

    paths = get_paths(dates, location_id=location_id, provider=provider)
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
