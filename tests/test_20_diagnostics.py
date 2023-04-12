import numpy as np
import xarray as xr

from c3s_eqc_automatic_quality_control import diagnostics


def test_grid_cell_area() -> None:
    earth_radius_m = 6_371e3
    ds = xr.Dataset(
        {
            "longitude": xr.DataArray(range(-180, 180), dims="longitude"),
            "latitude": xr.DataArray(range(-90, 90), dims="latitude"),
        }
    )
    ds = ds.cf.guess_coord_axis()
    actual = diagnostics.grid_cell_area(ds, earth_radius_m).sum()
    expected = 4 * np.pi * earth_radius_m**2
    assert np.isclose(actual, expected, rtol=1.0e-4)
