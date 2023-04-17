import cacholote
import numpy as np
import pyproj
import xarray as xr

from c3s_eqc_automatic_quality_control import diagnostics


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
