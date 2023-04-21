from typing import Any

import cacholote
import pyproj
import shapely
import xarray as xr


def poly_area(lon_bounds: list[Any], lat_bounds: list[Any], geod: pyproj.Geod) -> float:
    if len(lon_bounds) == len(lat_bounds) == 2:
        lon_bounds = sorted(lon_bounds) + sorted(lon_bounds, reverse=True)
        lat_bounds = [lat for lat in lat_bounds for _ in range(2)]
    polygon = shapely.Polygon(zip(lon_bounds, lat_bounds))
    return abs(geod.geometry_area_perimeter(polygon)[0])


@cacholote.cacheable
def cached_grid_cell_area(
    lon_bounds: dict[str, Any],
    lat_bounds: dict[str, Any],
    geod: pyproj.Geod,
) -> xr.Dataset:
    bounds_dim = set(lon_bounds["dims"]) & set(set(lat_bounds["dims"]))
    area = xr.apply_ufunc(
        poly_area,
        xr.DataArray.from_dict(lon_bounds),
        xr.DataArray.from_dict(lat_bounds),
        input_core_dims=[tuple(bounds_dim) for _ in range(2)],
        kwargs={"geod": geod},
        vectorize=True,
    )
    area.attrs["standard_name"] = "cell_area"
    cf_area: xr.DataArray = area.cf.add_canonical_attributes()
    return cf_area.to_dataset(name="cell_area")
