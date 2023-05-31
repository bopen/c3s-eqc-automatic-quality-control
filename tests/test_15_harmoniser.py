import numpy as np
import pytest
import xarray as xr

from c3s_eqc_automatic_quality_control import download


def test_swap_time_dim() -> None:
    ds = xr.DataArray(range(10), dims=("sounding_dim"), name="time").to_dataset()
    harmonised = download.harmonise(ds, "dummy_collection_id")
    assert set(harmonised.dims) == {"time"}


@pytest.mark.parametrize("time", ["time", "forecast_reference_time"])
def test_expand_time_dim(time: str) -> None:
    ds = xr.DataArray(range(10), dims=("dummy_dim"), name="dummy").to_dataset()
    ds[time] = xr.DataArray(0)
    harmonised = download.harmonise(ds, "dummy_collection_id")
    assert set(harmonised.dims) == {time, "dummy_dim"}


def test_coord_pressure() -> None:
    ds = xr.DataArray(range(10), dims=("pressure"), name="dummy").to_dataset()
    pre = xr.DataArray(range(10), dims=("pressure")) * 0.1
    ds["pre"] = pre
    harmonised = download.harmonise(ds, "dummy_collection_id")
    assert "pre" not in harmonised

    expected = harmonised["pressure"].values
    actual = pre.values
    np.testing.assert_equal(expected, actual)


def test_coord_bounds() -> None:
    ds = xr.Dataset(
        {
            "longitude": xr.DataArray(range(10), dims=("longitude")),
            "longitude_bounds": xr.DataArray(
                np.random.randn(10, 2),
                dims=("longitude", "bounds"),
                attrs={"standard_name": "longitude"},
            ),
        }
    )
    harmonised = download.harmonise(ds, "dummy_collection_id")
    assert "longitude_bounds" in harmonised.coords
    assert harmonised.cf.bounds["longitude"] == ["longitude_bounds"]
