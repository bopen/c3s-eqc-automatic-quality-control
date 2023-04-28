import xarray as xr

from c3s_eqc_automatic_quality_control import download


def test_swap_time_dim() -> None:
    ds = xr.DataArray(range(10), dims=("sounding_dim"), name="time").to_dataset()
    harmonised = download.harmonise(ds, "dummy_collection_id")
    assert set(harmonised.dims) == {"time"}


def test_expand_time_dim() -> None:
    ds = xr.DataArray(range(10), dims=("dummy_dim"), name="dummy").to_dataset()
    ds["time"] = xr.DataArray(0)
    harmonised = download.harmonise(ds, "dummy_collection_id")
    assert set(harmonised.dims) == {"time", "dummy_dim"}
