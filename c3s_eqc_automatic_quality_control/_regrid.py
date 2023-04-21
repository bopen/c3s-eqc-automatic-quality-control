from typing import Any

import cacholote
import fsspec
import xarray as xr
import xesmf as xe


@cacholote.cacheable
def regridder_weights(
    dict_in: dict[str, Any], dict_out: dict[str, Any], method: str, **kwargs: Any
) -> fsspec.spec.AbstractBufferedFile:
    ds_in = xr.Dataset.from_dict(dict_in)
    ds_out = xr.Dataset.from_dict(dict_out)
    regridder = xe.Regridder(ds_in, ds_out, method, **kwargs)
    with fsspec.open(regridder.to_netcdf(), "rb") as f:
        return f


def grid_to_dict(grid: xr.Dataset | xr.DataArray) -> dict[str, Any]:
    if isinstance(grid, xr.DataArray):
        grid = grid.to_dataset(name=grid.name or "__grid__")
    grid = grid.drop_vars((var for var, da in grid.variables.items() if not da.dims))
    coords = set()
    for coord in ("longitude", "latitude"):
        coords |= set(grid.cf.coordinates[coord] + grid.cf.bounds.get(coord, []))
    return xr.Dataset({coord: grid[coord] for coord in coords}).to_dict()


def cached_regridder(
    grid_in: xr.Dataset | xr.DataArray,
    grid_out: xr.Dataset | xr.DataArray,
    method: str,
    **kwargs: Any
) -> xe.Regridder:
    dict_in = grid_to_dict(grid_in)
    dict_out = grid_to_dict(grid_out)
    with cacholote.config.set(return_cache_entry=True, io_delete_original=True):
        cache_entry = regridder_weights(dict_in, dict_out, method, **kwargs)
    kwargs["weights"] = cache_entry.result["args"][0]["href"]
    return xe.Regridder(grid_in, grid_out, method, **kwargs)
