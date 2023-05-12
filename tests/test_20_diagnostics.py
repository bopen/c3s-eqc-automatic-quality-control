from typing import Union

import cacholote
import numpy as np
import pyproj
import pytest
import sklearn.metrics
import xarray as xr
from xarray.core.weighted import DataArrayWeighted, DatasetWeighted

from c3s_eqc_automatic_quality_control import diagnostics

XR_WEIGHTED_OR_NOT = Union[xr.DataArray, xr.Dataset, DataArrayWeighted, DatasetWeighted]


@pytest.mark.parametrize(
    "obj",
    [
        xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib"),
        xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib")["t2m"],
    ],
)
@pytest.mark.parametrize("weights", [True, False])
def test_spatial_weighted_statistics(
    obj: xr.DataArray | xr.Dataset, weights: bool
) -> None:
    weighted = obj.weighted(np.cos(np.deg2rad(obj["latitude"]))) if weights else obj

    expected_mean = weighted.mean(dim=("latitude", "longitude"))
    actual_mean = diagnostics.spatial_weighted_mean(obj, weights=weights)
    xr.testing.assert_equal(actual_mean, expected_mean)

    expected_std = weighted.std(dim=("latitude", "longitude"))
    actual_std = diagnostics.spatial_weighted_std(obj, weights=weights)
    xr.testing.assert_equal(actual_std, expected_std)

    expected_median = weighted.quantile(q=0.5, dim=("latitude", "longitude"))
    actual_median = diagnostics.spatial_weighted_median(obj, weights=weights)
    xr.testing.assert_equal(actual_median, expected_median)

    ds = xr.merge(
        [
            expected_mean.expand_dims(diagnostic=["mean"]),
            expected_median.expand_dims(diagnostic=["median"]),
            expected_std.expand_dims(diagnostic=["std"]),
        ],
    )
    expected_statistics = ds if isinstance(obj, xr.Dataset) else ds["t2m"]
    actual_statistics = diagnostics.spatial_weighted_statistics(obj, weights=weights)
    xr.testing.assert_equal(expected_statistics, actual_statistics)


@pytest.mark.parametrize(
    "obj",
    [
        xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib"),
        xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib")["t2m"],
    ],
)
@pytest.mark.parametrize("weights", [True, False])
def test_spatial_weighted_errors(obj: xr.DataArray | xr.Dataset, weights: bool) -> None:
    # Define all variables for equations
    obj1 = obj
    obj2 = obj**2
    diff2 = (obj2 - obj1) ** 2

    if weights:
        da_weights = np.cos(np.deg2rad(obj["latitude"]))
        obj1_weighted: XR_WEIGHTED_OR_NOT = obj1.weighted(da_weights)
        obj2_weighted: XR_WEIGHTED_OR_NOT = obj2.weighted(da_weights)
        diff2_weighted: XR_WEIGHTED_OR_NOT = diff2.weighted(da_weights)
    else:
        obj1_weighted = obj1
        obj2_weighted = obj2
        diff2_weighted = diff2

    mean1 = obj1_weighted.mean(dim=("latitude", "longitude"))
    mean2 = obj2_weighted.mean(dim=("latitude", "longitude"))
    std1 = obj1_weighted.std(dim=("latitude", "longitude"))
    std2 = obj2_weighted.std(dim=("latitude", "longitude"))
    diffc2 = ((obj2 - mean2) - (obj1 - mean1)) ** 2
    prod = (obj1 - mean1) * (obj2 - mean2)

    if weights:
        diffc2_weighted: XR_WEIGHTED_OR_NOT = diffc2.weighted(da_weights)
        prod_weighted: XR_WEIGHTED_OR_NOT = prod.weighted(da_weights)
    else:
        diffc2_weighted = diffc2
        prod_weighted = prod

    expected_rmse = diff2_weighted.mean(dim=("latitude", "longitude")) ** 0.5
    actual_rmse = diagnostics.spatial_weighted_rmse(obj1, obj2, weights=weights)
    xr.testing.assert_equal(expected_rmse, actual_rmse)

    expected_crmse = diffc2_weighted.mean(dim=("latitude", "longitude")) ** 0.5
    actual_crmse = diagnostics.spatial_weighted_crmse(obj1, obj2, weights=weights)
    xr.testing.assert_equal(expected_crmse, actual_crmse)

    expected_corr = prod_weighted.mean(dim=("latitude", "longitude")) / (std1 * std2)
    actual_corr = diagnostics.spatial_weighted_corr(obj1, obj2, weights=weights)
    xr.testing.assert_equal(expected_corr, actual_corr)

    ds = xr.merge(
        [
            expected_corr.expand_dims(diagnostic=["corr"]),
            expected_crmse.expand_dims(diagnostic=["crmse"]),
            expected_rmse.expand_dims(diagnostic=["rmse"]),
        ],
    )
    expected_errors = ds if isinstance(obj, xr.Dataset) else ds["t2m"]
    actual_errors = diagnostics.spatial_weighted_errors(obj1, obj2, weights=weights)
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


@pytest.mark.parametrize(
    "obj",
    [
        xr.tutorial.open_dataset("rasm"),
        xr.tutorial.open_dataset("rasm")["Tair"],
    ],
)
def test_time_weighted_mean(obj: xr.DataArray | xr.Dataset) -> None:
    weights = obj["time"].dt.days_in_month
    expected = (obj * weights).sum("time") / weights.sum("time")
    actual = diagnostics.time_weighted_mean(obj)
    xr.testing.assert_equal(expected, actual.fillna(0))


@pytest.mark.parametrize(
    "obj",
    [
        xr.tutorial.open_dataset("rasm"),
        xr.tutorial.open_dataset("rasm")["Tair"],
    ],
)
def test_seasonal_weighted_mean(obj: xr.DataArray | xr.Dataset) -> None:
    days_in_month = obj["time"].dt.days_in_month
    weights = (
        days_in_month.groupby("time.season")
        / days_in_month.groupby("time.season").sum()
    )
    expected = (obj * weights).groupby("time.season").sum(dim="time")
    actual = diagnostics.seasonal_weighted_mean(obj)
    xr.testing.assert_allclose(expected, actual.fillna(0))


@pytest.mark.parametrize(
    "obj",
    [
        xr.tutorial.open_dataset("rasm"),
        xr.tutorial.open_dataset("rasm")["Tair"],
    ],
)
def test_annual_weighted_mean(obj: xr.DataArray | xr.Dataset) -> None:
    days_in_month = obj["time"].dt.days_in_month
    weights = (
        days_in_month.groupby("time.year") / days_in_month.groupby("time.year").sum()
    )
    expected = (obj * weights).groupby("time.year").sum(dim="time")
    actual = diagnostics.annual_weighted_mean(obj)
    xr.testing.assert_allclose(expected, actual.fillna(0))


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
