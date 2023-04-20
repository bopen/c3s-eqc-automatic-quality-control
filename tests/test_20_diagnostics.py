import cacholote
import numpy as np
import pyproj
import pytest
import sklearn.metrics
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


@pytest.mark.parametrize(
    "obj",
    [
        xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib"),
        xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib")["t2m"],
    ],
)
def test_spatial_weighted_errors(obj: xr.DataArray | xr.Dataset) -> None:
    weights = np.cos(np.deg2rad(obj["latitude"]))

    obj1 = obj
    obj2 = obj**2

    mean1 = obj1.weighted(weights).mean(dim=("latitude", "longitude"))
    mean2 = obj2.weighted(weights).mean(dim=("latitude", "longitude"))
    std1 = obj1.weighted(weights).std(dim=("latitude", "longitude"))
    std2 = obj2.weighted(weights).std(dim=("latitude", "longitude"))

    expected_rmse = ((obj2 - obj1) ** 2).weighted(weights).mean(
        dim=("latitude", "longitude")
    ) ** 0.5
    actual_rmse = diagnostics.spatial_weighted_rmse(obj1, obj2)
    xr.testing.assert_equal(expected_rmse, actual_rmse)

    expected_crmse = (((obj2 - mean2) - (obj1 - mean1)) ** 2).weighted(weights).mean(
        dim=("latitude", "longitude")
    ) ** 0.5

    actual_crmse = diagnostics.spatial_weighted_crmse(obj1, obj2)
    xr.testing.assert_equal(expected_crmse, actual_crmse)

    expected_corr = ((obj1 - mean1) * (obj2 - mean2)).weighted(weights).mean(
        dim=("latitude", "longitude")
    ) / (std1 * std2)
    actual_corr = diagnostics.spatial_weighted_corr(obj1, obj2)
    xr.testing.assert_equal(expected_corr, actual_corr)

    expected_errors = xr.concat(
        [
            expected_rmse.expand_dims(diagnostics=["rmse"]),
            expected_crmse.expand_dims(diagnostics=["crmse"]),
            expected_corr.expand_dims(diagnostics=["corr"]),
        ],
        "diagnostics",
    )
    actual_errors = diagnostics.spatial_weighted_errors(obj1, obj2)
    xr.testing.assert_equal(expected_errors, actual_errors)


def test_spatial_weighted_rmse_against_sklearn() -> None:
    ds = xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib")
    da = ds["t2m"].isel(time=0, longitude=0)
    weights = np.cos(np.deg2rad(da["latitude"]))

    da1 = da
    da2 = da**2
    expected = sklearn.metrics.mean_squared_error(
        da1, da2, sample_weight=weights, squared=False
    )
    actual = diagnostics.spatial_weighted_rmse(da1, da2)
    assert expected == actual.values


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
