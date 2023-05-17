import pytest
import xarray as xr

from c3s_eqc_automatic_quality_control import utils


@pytest.mark.parametrize(
    "lon_slice,lat_slice,lon_out,lat_out",
    [
        [slice(-150, -140), slice(0, 10), range(-150, -139), range(0, 11)],
        [slice(-140, -150), slice(10, 0), range(-140, -151, -1), range(10, -1, -1)],
        [slice(320, 330), slice(None, None), range(320, 331), range(-90, 91)],
        [slice(330, 320), slice(None, None), range(330, 319, -1), range(-90, 91)],
    ],
)
@pytest.mark.parametrize("longitude", [range(-180, 180), range(0, 360)])
def test_regionalise_1d(
    longitude: range, lon_slice: slice, lat_slice: slice, lon_out: slice, lat_out: slice
) -> None:
    ds = xr.Dataset(
        {
            "longitude": xr.DataArray(longitude, dims="longitude"),
            "latitude": xr.DataArray(range(-90, 91), dims="latitude"),
        }
    ).cf.guess_coord_axis()
    expected = xr.Dataset(
        {
            "longitude": xr.DataArray(lon_out, dims="longitude"),
            "latitude": xr.DataArray(lat_out, dims="latitude"),
        }
    ).cf.guess_coord_axis()
    actual = utils.regionalise(ds, lon_slice, lat_slice)
    xr.testing.assert_identical(actual, expected)
    assert dict(ds.sizes) == {"longitude": 360, "latitude": 181}


def test_regionalise_2d() -> None:
    ds = (
        xr.merge(
            xr.broadcast(
                xr.DataArray(range(-180, 180), dims="x", name="longitude"),
                xr.DataArray(range(-90, 91), dims="y", name="latitude"),
            )
        )
        .set_coords(("longitude", "latitude"))
        .cf.guess_coord_axis()
    )
    expected = (
        xr.merge(
            xr.broadcast(
                xr.DataArray(range(-150, -140), dims="x", name="longitude"),
                xr.DataArray(range(0, 10), dims="y", name="latitude"),
            )
        )
        .set_coords(("longitude", "latitude"))
        .cf.guess_coord_axis()
    )
    actual = utils.regionalise(ds, lon_slice=slice(-150, -140), lat_slice=slice(0, 10))
    xr.testing.assert_identical(actual, expected)
    assert dict(ds.sizes) == {"x": 360, "y": 181}
