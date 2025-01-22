from typing import overload

import numpy as np
import pytest
import xarray as xr
import xskillscore as xs
from packaging.version import Version

from c3s_eqc_automatic_quality_control import diagnostics

WEIGHTED_POLYFIT_IS_BROKEN = Version(xr.__version__) >= Version("v2024.11.0")


@overload
def weighted_mean(obj: xr.DataArray) -> xr.DataArray: ...


@overload
def weighted_mean(obj: xr.Dataset) -> xr.Dataset: ...


def weighted_mean(obj: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
    da_weights = obj["time"].dt.days_in_month
    obj = (obj * da_weights).sum("time") / da_weights.sum("time")
    return obj


@overload
def weighted_std(obj: xr.DataArray) -> xr.DataArray: ...


@overload
def weighted_std(obj: xr.Dataset) -> xr.Dataset: ...


def weighted_std(obj: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
    da_weights = obj["time"].dt.days_in_month
    w_mean = weighted_mean(obj)
    obj = np.sqrt(
        (da_weights * (obj - w_mean) ** 2).sum("time") / da_weights.sum("time")
    )
    return obj


@pytest.mark.parametrize(
    "obj",
    [
        # xr.tutorial.open_dataset("rasm"),
        xr.tutorial.open_dataset("rasm")["Tair"],
    ],
)
@pytest.mark.parametrize("weights", [True, False])
class TestTimeWeighted:
    def test_time_weighted_mean(
        self, obj: xr.DataArray | xr.Dataset, weights: bool
    ) -> None:
        if weights:
            expected = weighted_mean(obj)
            expected = expected.where(expected != 0)
        else:
            expected = obj.mean("time")
        actual = diagnostics.time_weighted_mean(obj, weights=weights)
        xr.testing.assert_allclose(expected, actual)

    def test_time_weighted_std(
        self, obj: xr.DataArray | xr.Dataset, weights: bool
    ) -> None:
        if weights:
            expected = weighted_std(obj)
            expected = expected.where(expected != 0)
        else:
            expected = obj.std("time")
        actual = diagnostics.time_weighted_std(obj, weights=weights)
        xr.testing.assert_allclose(expected, actual)

    def test_monthly_weighted_mean(
        self, obj: xr.DataArray | xr.Dataset, weights: bool
    ) -> None:
        if weights:
            expected = (obj).groupby("time.month").map(weighted_mean)
            expected = expected.where(expected != 0)
        else:
            expected = obj.groupby("time.month").mean("time")
        actual = diagnostics.monthly_weighted_mean(obj, weights=weights)
        xr.testing.assert_allclose(expected, actual)

    def test_seasonal_weighted_mean(
        self, obj: xr.DataArray | xr.Dataset, weights: bool
    ) -> None:
        if weights:
            expected = (obj).groupby("time.season").map(weighted_mean)
            expected = expected.where(expected != 0)
        else:
            expected = obj.groupby("time.season").mean("time")
        actual = diagnostics.seasonal_weighted_mean(obj, weights=weights)
        xr.testing.assert_allclose(expected, actual)

    def test_monthly_weighted_std(
        self, obj: xr.DataArray | xr.Dataset, weights: bool
    ) -> None:
        if weights:
            expected = obj.groupby("time.month").map(weighted_std)
            expected = expected.where(expected != 0)
        else:
            expected = obj.groupby("time.month").std("time")
        actual = diagnostics.monthly_weighted_std(obj, weights=weights)
        xr.testing.assert_allclose(expected, actual)

    def test_seasonal_weighted_std(
        self, obj: xr.DataArray | xr.Dataset, weights: bool
    ) -> None:
        if weights:
            expected = obj.groupby("time.season").map(weighted_std)
            expected = expected.where(expected != 0)
        else:
            expected = obj.groupby("time.season").std("time")
        actual = diagnostics.seasonal_weighted_std(obj, weights=weights)
        xr.testing.assert_allclose(expected, actual)

    def test_annual_weighted_mean(
        self, obj: xr.DataArray | xr.Dataset, weights: bool
    ) -> None:
        if weights:
            expected = (obj).groupby("time.year").map(weighted_mean)
            expected = expected.where(expected != 0)
        else:
            expected = obj.groupby("time.year").mean("time")
        actual = diagnostics.annual_weighted_mean(obj, weights=weights)
        xr.testing.assert_allclose(expected, actual)

    def test_annual_weighted_std(
        self, obj: xr.DataArray | xr.Dataset, weights: bool
    ) -> None:
        if weights:
            if WEIGHTED_POLYFIT_IS_BROKEN:
                pytest.xfail("See https://github.com/pydata/xarray/issues/9972")
            expected = (obj).groupby("time.year").map(weighted_std)
            expected = expected.where(expected != 0)
        else:
            expected = obj.groupby("time.year").std("time")
        actual = diagnostics.annual_weighted_std(obj, weights=weights)
        xr.testing.assert_allclose(expected, actual)

    def test_time_weighted_linear_trend(
        self, obj: xr.DataArray | xr.Dataset, weights: bool
    ) -> None:
        if weights and WEIGHTED_POLYFIT_IS_BROKEN:
            pytest.xfail("See https://github.com/pydata/xarray/issues/9972")

        actual = diagnostics.time_weighted_linear_trend(obj, weights=weights)

        ds_trend = (
            obj.polyfit(
                "time", w=obj["time"].dt.days_in_month if weights else None, deg=1
            ).sel(degree=1, drop=True)
            * 1.0e9
        )
        if isinstance(obj, xr.DataArray):
            xr.testing.assert_equal(ds_trend["polyfit_coefficients"], actual)
        else:
            xr.testing.assert_equal(
                ds_trend.rename(Tair_polyfit_coefficients="Tair"), actual
            )

    def test_time_weighted_linear_trend_stats(
        self, obj: xr.DataArray | xr.Dataset, weights: bool
    ) -> None:
        if weights and WEIGHTED_POLYFIT_IS_BROKEN:
            pytest.xfail("See https://github.com/pydata/xarray/issues/9972")

        actual_dict = diagnostics.time_weighted_linear_trend(
            obj, weights=weights, p_value=True, r2=True
        )
        assert isinstance(actual_dict, dict)

        da_weights = obj["time"].dt.days_in_month if weights else None
        coeff = obj.polyfit(dim="time", deg=1, w=da_weights)
        if isinstance(obj, xr.DataArray):
            coeff = coeff.polyfit_coefficients
        else:
            coeff = coeff.rename(
                {
                    var: var.replace("_polyfit_coefficients", "")
                    for var in map(str, coeff.data_vars)
                }
            )
        fit = xr.polyval(obj["time"], coeff)
        expected_dict = {
            "linear_trend": diagnostics.time_weighted_linear_trend(
                obj, weights=weights
            ),
            "p_value": xs.pearson_r_p_value(obj, fit, "time", weights=da_weights),
            "r2": xs.r2(obj, fit, "time", weights=da_weights),
        }
        assert set(actual_dict) == set(expected_dict)
        for key in actual_dict:
            xr.testing.assert_identical(actual_dict[key], actual_dict[key])

    def test_time_weighted_coverage(
        self, obj: xr.DataArray | xr.Dataset, weights: bool
    ) -> None:
        if weights:
            expected = (
                obj.notnull()
                * obj["time"].dt.days_in_month
                / obj["time"].dt.days_in_month.sum("time")
            )
        else:
            expected = obj.count("time") / obj.sizes["time"]
        actual = diagnostics.time_weighted_coverage(obj, weights=weights)
        xr.testing.assert_equal(expected, actual)
