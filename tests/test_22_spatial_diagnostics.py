import numpy as np
import pytest
import sklearn.metrics
import xarray as xr

try:
    from xarray.core.weighted import DataArrayWeighted, DatasetWeighted
except ImportError:
    from xarray.computation.weighted import DataArrayWeighted, DatasetWeighted

from c3s_eqc_automatic_quality_control import diagnostics

XR_WEIGHTED_OR_NOT = xr.DataArray | xr.Dataset | DataArrayWeighted | DatasetWeighted


@pytest.mark.parametrize(
    "obj",
    [
        xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib"),
        xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib")["t2m"],
    ],
)
@pytest.mark.parametrize("weights", [True, False])
class TestSpatialWeighted:
    def test_spatial_weighted_statistics(
        self, obj: xr.DataArray | xr.Dataset, weights: bool
    ) -> None:
        weighted = obj.weighted(np.cos(np.deg2rad(obj["latitude"]))) if weights else obj

        expected_mean = weighted.mean(dim=("latitude", "longitude"))
        actual_mean = diagnostics.spatial_weighted_mean(obj, weights=weights)
        xr.testing.assert_equal(actual_mean, expected_mean)

        expected_std = weighted.std(dim=("latitude", "longitude"))
        actual_std = diagnostics.spatial_weighted_std(obj, weights=weights)
        xr.testing.assert_equal(actual_std, expected_std)

        expected_median = weighted.quantile(
            q=0.5, dim=("latitude", "longitude")
        ).drop_vars("quantile")
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
        actual_statistics = diagnostics.spatial_weighted_statistics(
            obj, weights=weights
        )
        xr.testing.assert_equal(expected_statistics, actual_statistics)

    def test_spatial_weighted_errors(
        self, obj: xr.DataArray | xr.Dataset, weights: bool
    ) -> None:
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

        expected_corr = prod_weighted.mean(dim=("latitude", "longitude")) / (
            std1 * std2
        )
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
    da = ds["t2m"].astype("float64").isel(time=0, longitude=0)
    weights = np.cos(np.deg2rad(da["latitude"]))

    da1 = da
    da2 = da**2
    expected = sklearn.metrics.root_mean_squared_error(da1, da2, sample_weight=weights)
    actual = diagnostics.spatial_weighted_rmse(da1, da2)
    assert expected == actual.values
