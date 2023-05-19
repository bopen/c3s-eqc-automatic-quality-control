import cacholote
import numpy as np
import pyproj
import pytest
import xarray as xr

from c3s_eqc_automatic_quality_control import diagnostics


@pytest.mark.parametrize(
    "obj",
    [
        xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib"),
        xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib")["t2m"],
    ],
)
def test_regrid(obj: xr.DataArray | xr.Dataset) -> None:
    fs, dirname = cacholote.utils.get_cache_files_fs_dirname()
    assert fs.ls(dirname) == []  # cache is empty

    for _, obj in obj.isel(time=slice(2)).groupby("time"):
        expected = obj.isel(longitude=slice(10), latitude=slice(10))
        actual = diagnostics.regrid(obj, expected, "nearest_s2d")
        xr.testing.assert_equal(actual, expected)
        assert actual.attrs["regrid_method"] == "nearest_s2d"
        assert len(fs.ls(dirname)) == 1  # re-use weights


def test_grid_cell_area() -> None:
    fs, dirname = cacholote.utils.get_cache_files_fs_dirname()
    assert fs.ls(dirname) == []  # cache is empty

    # Compute area spheroid
    geod = pyproj.Geod(ellps="WGS84")
    e = np.sqrt(2 * geod.f - geod.f**2)
    first = 2 * np.pi * geod.a**2
    second = (np.pi * geod.b**2 / e) * np.log((1 + e) / (1 - e))
    expected = first + second

    # Global coordinates
    ds = xr.Dataset(
        {
            "longitude": xr.DataArray(np.arange(-180, 180, 10), dims="longitude"),
            "latitude": xr.DataArray(np.arange(-90, 91, 2), dims="latitude"),
        }
    )
    ds = ds.cf.guess_coord_axis()
    da = ds["longitude"] * ds["latitude"]

    for obj in (ds, da):
        actual = diagnostics.grid_cell_area(obj).sum().values
        np.testing.assert_approx_equal(actual, expected, significant=4)
        assert len(fs.ls(dirname)) == 1  # re-use weights
