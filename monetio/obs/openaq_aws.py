"""OpenAQ archive data on AWS.

https://registry.opendata.aws/openaq/
https://docs.openaq.org/docs/accessing-openaq-archive-data
"""

import pandas as pd


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
