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
            "longitude": xr.DataArray(range(-180, 180), dims="longitude"),
            "latitude": xr.DataArray(range(-90, 90), dims="latitude"),
        }
    )
    ds = ds.cf.guess_coord_axis()
    with cacholote.config.set(use_cache=False):
        actual = diagnostics.grid_cell_area(ds).sum().values

    assert np.isclose(actual, expected, rtol=1.0e-4)
