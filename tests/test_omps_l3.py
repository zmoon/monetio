from pathlib import Path

import pandas as pd

from monetio.sat._omps_l3_mm import open_dataset


def test_open_dataset():
    ps = sorted(Path("~/Downloads/").expanduser().glob("OMPS-TO3-L3-DAILY_v2.1_*.h5"))
    ds = open_dataset(ps)

    assert ds.sizes["time"] == len(ps) == 2
    assert ds.sizes["x"] == 360
    assert ds.sizes["y"] == 180

    assert ds.time.isel(time=0) == pd.Timestamp("2019-09-05")
    assert ds.time.isel(time=1) == pd.Timestamp("2019-09-06")
    assert ds.longitude.dims == ds.latitude.dims == ("y", "x")
    assert ds.longitude.min() == -179.5
    assert ds.longitude.max() == 179.5
    assert ds.latitude.min() == -89.5
    assert ds.latitude.max() == 89.5

    assert set(ds.data_vars) == {"ozone_column"}
    assert 150 < ds.ozone_column.mean() < 400
    assert ds.ozone_column.min() > 100
    assert ds.ozone_column.max() < 600
