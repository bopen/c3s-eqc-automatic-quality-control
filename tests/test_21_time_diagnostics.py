from typing import overload

import numpy as np
import pytest
import xarray as xr

from c3s_eqc_automatic_quality_control import diagnostics


@overload
def weighted_mean(obj: xr.DataArray) -> xr.DataArray:
    ...


@overload
def weighted_mean(obj: xr.Dataset) -> xr.Dataset:
    ...


def weighted_mean(obj: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
    da_weights = obj["time"].dt.days_in_month
    obj = (obj * da_weights).sum("time") / da_weights.sum("time")
    return obj


@overload
def weighted_std(obj: xr.DataArray) -> xr.DataArray:
    ...


@overload
def weighted_std(obj: xr.Dataset) -> xr.Dataset:
    ...


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
        xr.tutorial.open_dataset("rasm"),
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
        xr.testing.assert_equal(expected, actual)

    def test_time_weighted_std(
        self, obj: xr.DataArray | xr.Dataset, weights: bool
    ) -> None:
        if weights:
            expected = weighted_std(obj)
            expected = expected.where(expected != 0)
        else:
            expected = obj.std("time")
        actual = diagnostics.time_weighted_std(obj, weights=weights)
        xr.testing.assert_equal(expected, actual)

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
            expected = (obj).groupby("time.year").map(weighted_std)
            expected = expected.where(expected != 0)
        else:
            expected = obj.groupby("time.year").std("time")
        actual = diagnostics.annual_weighted_std(obj, weights=weights)
        xr.testing.assert_allclose(expected, actual)
