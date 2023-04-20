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
def test_spatial_weighted_statistics(obj: xr.DataArray | xr.Dataset) -> None:
    weighted = obj.weighted(np.cos(np.deg2rad(obj["latitude"])))

    expected_mean = weighted.mean(dim=("latitude", "longitude"))
    actual_mean = diagnostics.spatial_weighted_mean(obj)
    xr.testing.assert_equal(actual_mean, expected_mean)

    expected_std = weighted.std(dim=("latitude", "longitude"))
    actual_std = diagnostics.spatial_weighted_std(obj)
    xr.testing.assert_equal(actual_std, expected_std)

    expected_median = weighted.quantile(q=0.5, dim=("latitude", "longitude"))
    actual_median = diagnostics.spatial_weighted_median(obj)
    xr.testing.assert_equal(actual_median, expected_median)

    expected_statistics = xr.concat(
        [
            expected_mean.expand_dims(diagnostics=["mean"]),
            expected_std.expand_dims(diagnostics=["std"]),
            expected_median.expand_dims(diagnostics=["median"]),
        ],
        "diagnostics",
    )
    actual_statistics = diagnostics.spatial_weighted_statistics(obj)
    xr.testing.assert_equal(expected_statistics, actual_statistics)


def test_grid_cell_area() -> None:
    geod = pyproj.Geod(ellps="WGS84")

    # Compute area spheroid
    e = np.sqrt(2 * geod.f - geod.f**2)
    first = 2 * np.pi * geod.a**2
    second = (np.pi * geod.b**2 / e) * np.log((1 + e) / (1 - e))
    expected = first + second

    # Get area
    ds = xr.Dataset(
        {
            "longitude": xr.DataArray(np.arange(-180, 180, 10), dims="longitude"),
            "latitude": xr.DataArray(np.arange(-90, 90, 2), dims="latitude"),
        }
    )
    ds = ds.cf.guess_coord_axis()
    with cacholote.config.set(use_cache=False):
        actual = diagnostics.grid_cell_area(ds).sum().values

    np.testing.assert_approx_equal(actual, expected, significant=4)
